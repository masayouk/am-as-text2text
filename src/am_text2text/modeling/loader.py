import gc
from pathlib import Path
from typing import Optional

import torch
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)

from am_text2text.schemas.annotator import AnnotatorModelConfig


def reset_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def load_model_and_tokenizer(
    model_name_or_path: str,
    *,
    quantization_mode: str,
    quant_type: str,
    use_double_quant: bool,
    cache_dir: Optional[str] = None,
    is_inference: bool = False,
    local_rank: Optional[int] = None,
):
    model_config = AutoConfig.from_pretrained(model_name_or_path, cache_dir=cache_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, cache_dir=cache_dir)
    model_dtype = _model_dtype()
    quantization_config = _build_quantization_config(
        quantization_mode=quantization_mode,
        quant_type=quant_type,
        use_double_quant=use_double_quant,
        model_dtype=model_dtype,
    )

    device_map = _resolve_device_map(local_rank=local_rank)

    load_kwargs = {
        "config": model_config,
        "cache_dir": cache_dir,
        "torch_dtype": model_dtype,
        "device_map": device_map,
    }
    if quantization_config is not None:
        load_kwargs["quantization_config"] = quantization_config

    if model_config.is_encoder_decoder:
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path, **load_kwargs)
        tokenizer.padding_side = "right"
    else:
        model = AutoModelForCausalLM.from_pretrained(model_name_or_path, **load_kwargs)
        tokenizer.padding_side = "left" if is_inference else "right"
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
    return model_config, tokenizer, model


def create_adapter_model(base_model, model_config, model_options: AnnotatorModelConfig):
    if not model_options.adapter.enabled:
        return base_model

    try:
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("peft is required when model.adapter.enabled=true") from exc

    prepared_model = base_model
    if model_options.quantization.mode == "4bit":
        prepared_model = prepare_model_for_kbit_training(base_model)

    modules = find_all_linear_names(prepared_model)
    if not modules:
        raise ValueError("No linear modules were found for LoRA target_modules")
    task_type = "SEQ_2_SEQ_LM" if model_config.is_encoder_decoder else "CAUSAL_LM"
    lora_config = LoraConfig(
        r=model_options.adapter.r,
        lora_alpha=model_options.adapter.alpha,
        lora_dropout=model_options.adapter.dropout,
        bias="none",
        task_type=task_type,
        target_modules=modules,
    )
    return get_peft_model(prepared_model, lora_config)


def load_inference_model(
    *,
    base_model_name: str,
    checkpoint_path: str,
    model_options: AnnotatorModelConfig,
    cache_dir: Optional[str] = None,
):
    checkpoint_path_str = str(checkpoint_path)
    adapter_config_path = Path(checkpoint_path_str) / "adapter_config.json"
    if adapter_config_path.exists():
        try:
            from peft import PeftModel
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("peft is required to load LoRA or QLoRA checkpoints") from exc
        model_config, tokenizer, base_model = load_model_and_tokenizer(
            base_model_name,
            quantization_mode=model_options.quantization.mode,
            quant_type=model_options.quantization.quant_type,
            use_double_quant=model_options.quantization.use_double_quant,
            cache_dir=cache_dir,
            is_inference=True,
        )
        model = PeftModel.from_pretrained(base_model, checkpoint_path_str, is_trainable=False)
        return model_config, tokenizer, model

    model_config, tokenizer, model = load_model_and_tokenizer(
        checkpoint_path_str,
        quantization_mode="none",
        quant_type=model_options.quantization.quant_type,
        use_double_quant=model_options.quantization.use_double_quant,
        cache_dir=cache_dir,
        is_inference=True,
    )
    return model_config, tokenizer, model


def find_all_linear_names(model) -> list[str]:
    target_types: tuple[type, ...] = (torch.nn.Linear,)
    try:
        import bitsandbytes as bnb

        target_types = target_types + (bnb.nn.Linear4bit, bnb.nn.Linear8bitLt)
    except ImportError:
        pass

    lora_module_names: set[str] = set()
    for name, module in model.named_modules():
        if isinstance(module, target_types):
            names = name.split(".")
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    lora_module_names.discard("lm_head")
    return sorted(lora_module_names)


def _build_quantization_config(
    *,
    quantization_mode: str,
    quant_type: str,
    use_double_quant: bool,
    model_dtype,
):
    if quantization_mode != "4bit":
        return None
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type=quant_type,
        bnb_4bit_use_double_quant=use_double_quant,
        bnb_4bit_compute_dtype=model_dtype,
    )


def _resolve_device_map(*, local_rank: Optional[int]):
    if local_rank is not None and local_rank >= 0:
        return {"": local_rank}
    if not torch.cuda.is_available():
        return None
    return {"": 0}


def _model_dtype():
    if not torch.cuda.is_available():
        return torch.float32
    if torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16
