import argparse
import json
import os
from pathlib import Path
from typing import Optional

import torch
from pydantic import BaseModel
from transformers import Trainer, TrainingArguments, default_data_collator, set_seed

from am_text2text.modeling.loader import create_adapter_model, load_model_and_tokenizer, reset_memory
from am_text2text.schemas.annotator import AnnotatorModelConfig, TrainConfig
from am_text2text.training.dataset_loader import Seq2SeqAndCausalDatasetLoader
from am_text2text.utils import setup_logging


class TrainWorkerRequest(BaseModel):
    data_dir: Path
    dataset_name: str
    output_dir: Path
    model: AnnotatorModelConfig
    train: TrainConfig
    seed: int = 42
    cache_dir: Optional[Path] = None
    resume_from_checkpoint: Optional[Path] = None


class AnnotatorTrainer(Trainer):
    def _save(self, output_dir: str | None = None, state_dict: dict | None = None) -> None:
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)

        model_to_save = self.accelerator.unwrap_model(self.model, keep_torch_compile=False)
        try:
            from peft import PeftModel
        except ImportError:  # pragma: no cover
            PeftModel = None

        if PeftModel is not None and isinstance(model_to_save, PeftModel):
            if state_dict is None:
                state_dict = self.model.state_dict()
            model_to_save.save_pretrained(output_dir, state_dict=state_dict, save_embedding_layers=False)
            if self.processing_class is not None:
                self.processing_class.save_pretrained(output_dir)
            elif (
                self.data_collator is not None
                and hasattr(self.data_collator, "tokenizer")
                and self.data_collator.tokenizer is not None
            ):
                self.data_collator.tokenizer.save_pretrained(output_dir)
            torch.save(self.args, os.path.join(output_dir, "training_args.bin"))
            return

        super()._save(output_dir, state_dict)


def run_train_request(request: TrainWorkerRequest) -> Path:
    setup_logging()
    request.output_dir.mkdir(parents=True, exist_ok=True)

    visible_devices = request.train.gpu.visible_devices_env()
    if visible_devices:
        os.environ["CUDA_VISIBLE_DEVICES"] = visible_devices

    local_rank = int(os.environ.get("LOCAL_RANK", "-1"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    set_seed(request.seed)

    model_config, tokenizer, model = load_model_and_tokenizer(
        request.model.base_model,
        quantization_mode=request.model.quantization.mode if request.model.adapter.enabled else "none",
        quant_type=request.model.quantization.quant_type,
        use_double_quant=request.model.quantization.use_double_quant,
        cache_dir=str(request.cache_dir) if request.cache_dir is not None else None,
        is_inference=False,
        local_rank=local_rank if world_size > 1 else None,
    )
    model = create_adapter_model(model, model_config, request.model)

    loader = Seq2SeqAndCausalDatasetLoader(
        data_dir=request.data_dir,
        tokenizer=tokenizer,
        max_input_length=request.train.max_seq_length,
        max_output_length=request.train.max_seq_length,
        prompt_dir=request.model.prompt.prompt_dir if not model_config.is_encoder_decoder else None,
    )
    model_type = "seq2seq" if model_config.is_encoder_decoder else "causal"
    train_dataset = loader.load_dataset(
        dataset_name=request.dataset_name,
        split="train",
        model_type=model_type,
        prompt_name=request.model.prompt.prompt_name,
        system_prompt_name=request.model.prompt.system_prompt_name,
        use_chat_template=request.model.prompt.use_chat_template,
    )

    training_args = TrainingArguments(
        output_dir=str(request.output_dir),
        max_steps=request.train.max_steps,
        save_steps=request.train.save_steps,
        save_strategy="steps",
        learning_rate=request.train.learning_rate,
        per_device_train_batch_size=request.train.per_device_train_batch_size,
        per_device_eval_batch_size=request.train.per_device_eval_batch_size,
        gradient_accumulation_steps=request.train.gradient_accumulation_steps,
        warmup_steps=request.train.warmup_steps,
        logging_steps=request.train.logging_steps,
        lr_scheduler_type=request.train.lr_scheduler_type,
        bf16=request.train.bf16,
        optim=request.train.optim,
        report_to="none",
        remove_unused_columns=False,
        gradient_checkpointing=request.train.gradient_checkpointing,
        do_train=True,
        deepspeed=(
            str(request.train.deepspeed.config_path)
            if request.train.gpu.num_gpus > 1 and request.train.deepspeed.enabled and request.train.deepspeed.config_path is not None
            else None
        ),
    )

    trainer = AnnotatorTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=default_data_collator,
    )
    trainer.train(
        resume_from_checkpoint=str(request.resume_from_checkpoint)
        if request.resume_from_checkpoint is not None
        else None
    )
    trainer.save_model(str(request.output_dir))

    done_payload = {
        "status": "completed",
        "global_step": int(trainer.state.global_step),
        "world_size": world_size,
        "seed": request.seed,
    }
    (request.output_dir / "done.json").write_text(json.dumps(done_payload, indent=2), encoding="utf-8")

    del trainer
    del model
    del tokenizer
    del train_dataset
    reset_memory()
    return request.output_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", required=True)
    args = parser.parse_args()
    request = TrainWorkerRequest.model_validate_json(Path(args.request).read_text(encoding="utf-8"))
    run_train_request(request)


if __name__ == "__main__":
    main()
