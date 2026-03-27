import argparse
import json
from pathlib import Path
from typing import Optional

import torch
from pydantic import BaseModel
from torch.utils.data import DataLoader

from am_text2text.modeling.loader import (
    load_inference_model,
    load_model_and_tokenizer,
    reset_memory,
)
from am_text2text.schemas.annotator import AnnotatorModelConfig, DecodeConfig
from am_text2text.training.dataset_loader import Seq2SeqAndCausalDatasetLoader
from am_text2text.utils import setup_logging


class AnnotationRequest(BaseModel):
    data_dir: Path
    dataset_name: str
    checkpoint_path: Path
    output_dir: Path
    split: str
    model: AnnotatorModelConfig
    max_seq_length: int = 1024
    decode: DecodeConfig = DecodeConfig()
    seed: int = 42
    cache_dir: Optional[Path] = None


class ReusableAnnotationSession:
    def __init__(self, request: AnnotationRequest) -> None:
        if request.decode.gpu.num_gpus != 1:
            raise ValueError("ReusableAnnotationSession only supports single-GPU inference")
        if not _is_adapter_checkpoint(request.checkpoint_path):
            raise ValueError("ReusableAnnotationSession requires adapter checkpoints")

        self.request = request
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_config, self.tokenizer, self.model = load_model_and_tokenizer(
            request.model.base_model,
            quantization_mode=request.model.quantization.mode if request.model.adapter.enabled else "none",
            quant_type=request.model.quantization.quant_type,
            use_double_quant=request.model.quantization.use_double_quant,
            cache_dir=str(request.cache_dir) if request.cache_dir is not None else None,
            is_inference=True,
        )
        self.current_adapter_name: str | None = None
        self.current_checkpoint_path: Path | None = None
        self.dataset_cache: dict[str, torch.utils.data.Dataset] = {}

    def activate_checkpoint(self, checkpoint_path: Path) -> None:
        try:
            from peft import PeftModel
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("peft is required to load LoRA or QLoRA checkpoints") from exc

        checkpoint_path = checkpoint_path.resolve()
        if self.current_checkpoint_path == checkpoint_path:
            self.model.eval()
            return

        adapter_name = _adapter_name_for_checkpoint(checkpoint_path)
        if self.current_adapter_name is None:
            self.model = PeftModel.from_pretrained(
                self.model,
                str(checkpoint_path),
                adapter_name=adapter_name,
                is_trainable=False,
            )
        else:
            self.model.delete_adapter(self.current_adapter_name)
            self.model.load_adapter(str(checkpoint_path), adapter_name=adapter_name, is_trainable=False)
            self.model.set_adapter(adapter_name)
        self.current_adapter_name = adapter_name
        self.current_checkpoint_path = checkpoint_path
        self.model.eval()

    def load_dataset(self, split: str):
        if split not in self.dataset_cache:
            loader = Seq2SeqAndCausalDatasetLoader(
                data_dir=self.request.data_dir,
                tokenizer=self.tokenizer,
                max_input_length=self.request.max_seq_length,
                max_output_length=self.request.max_seq_length,
                prompt_dir=self.request.model.prompt.prompt_dir if not self.model_config.is_encoder_decoder else None,
            )
            self.dataset_cache[split] = loader.load_dataset(
                dataset_name=self.request.dataset_name,
                split=split,
                model_type="seq2seq" if self.model_config.is_encoder_decoder else "causal",
                prompt_name=self.request.model.prompt.prompt_name,
                system_prompt_name=self.request.model.prompt.system_prompt_name,
                is_inference=True,
                use_chat_template=self.request.model.prompt.use_chat_template,
            )
        return self.dataset_cache[split]

    def close(self) -> None:
        if hasattr(self, "model"):
            del self.model
        if hasattr(self, "tokenizer"):
            del self.tokenizer
        self.dataset_cache.clear()
        reset_memory()

def run_annotation_request(request: AnnotationRequest, session: ReusableAnnotationSession | None = None) -> Path:
    setup_logging()
    request.output_dir.mkdir(parents=True, exist_ok=True)
    if request.decode.gpu.num_gpus != 1:
        raise ValueError("Annotation currently supports only a single GPU")

    if request.decode.do_sample:
        torch.manual_seed(request.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(request.seed)

    if session is None:
        model_config, tokenizer, model = load_inference_model(
            base_model_name=request.model.base_model,
            checkpoint_path=str(request.checkpoint_path),
            model_options=request.model,
            cache_dir=str(request.cache_dir) if request.cache_dir is not None else None,
        )
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model.eval()

        loader = Seq2SeqAndCausalDatasetLoader(
            data_dir=request.data_dir,
            tokenizer=tokenizer,
            max_input_length=request.max_seq_length,
            max_output_length=request.max_seq_length,
            prompt_dir=request.model.prompt.prompt_dir if not model_config.is_encoder_decoder else None,
        )
        dataset = loader.load_dataset(
            dataset_name=request.dataset_name,
            split=request.split,
            model_type="seq2seq" if model_config.is_encoder_decoder else "causal",
            prompt_name=request.model.prompt.prompt_name,
            system_prompt_name=request.model.prompt.system_prompt_name,
            is_inference=True,
            use_chat_template=request.model.prompt.use_chat_template,
        )
    else:
        session.activate_checkpoint(request.checkpoint_path)
        model_config = session.model_config
        tokenizer = session.tokenizer
        model = session.model
        device = session.device
        dataset = session.load_dataset(request.split)
    dataloader = DataLoader(dataset, batch_size=request.decode.batch_size, shuffle=False)

    if model_config.is_encoder_decoder:
        generation_kwargs = {
            "max_length": request.decode.max_length,
            "min_length": request.decode.min_length,
            "num_beams": request.decode.num_beams,
            "length_penalty": request.decode.length_penalty,
            "no_repeat_ngram_size": request.decode.no_repeat_ngram_size,
            "early_stopping": request.decode.early_stopping,
        }
    else:
        generation_kwargs = {
            "max_new_tokens": request.decode.max_new_tokens,
            "num_beams": request.decode.num_beams,
            "temperature": request.decode.temperature,
            "top_k": request.decode.top_k,
            "top_p": request.decode.top_p,
            "do_sample": request.decode.do_sample,
        }

    predictions: list[str] = []
    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)
        with torch.no_grad():
            outputs = model.generate(input_ids=input_ids, attention_mask=attention_mask, **generation_kwargs)
        if model_config.is_encoder_decoder:
            decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        else:
            prompt_len = input_ids.shape[1]
            decoded = tokenizer.batch_decode(outputs[:, prompt_len:], skip_special_tokens=True)

        predictions.extend(text.replace("\n", "") for text in decoded)
    merged_path = request.output_dir / "raw_predictions.json"
    merged_path.write_text(
        json.dumps({"predictions": predictions}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    if session is None:
        del model
        del tokenizer
        reset_memory()
    return merged_path


def _is_adapter_checkpoint(checkpoint_path: Path) -> bool:
    return (checkpoint_path / "adapter_config.json").exists()


def _adapter_name_for_checkpoint(checkpoint_path: Path) -> str:
    return checkpoint_path.name.replace("-", "_")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", required=True)
    args = parser.parse_args()
    request = AnnotationRequest.model_validate_json(Path(args.request).read_text(encoding="utf-8"))
    run_annotation_request(request)


if __name__ == "__main__":
    main()
