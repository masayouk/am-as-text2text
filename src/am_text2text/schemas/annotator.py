from pathlib import Path
from typing import Literal, Optional

from pydantic import BaseModel, Field, model_validator

from am_text2text.schemas.common import DeepSpeedConfig, GpuConfig, SplitName


class PromptConfig(BaseModel):
    prompt_dir: Path = Path("prompts")
    prompt_name: Optional[str] = None
    system_prompt_name: Optional[str] = None
    use_chat_template: bool = True


class AdapterConfig(BaseModel):
    enabled: bool = True
    type: Literal["lora"] = "lora"
    r: int = 16
    alpha: int = 32
    dropout: float = 0.05


class QuantizationConfig(BaseModel):
    mode: Literal["none", "4bit"] = "4bit"
    quant_type: Literal["nf4", "fp4"] = "nf4"
    use_double_quant: bool = True


class AnnotatorModelConfig(BaseModel):
    base_model: str
    prompt: PromptConfig = Field(default_factory=PromptConfig)
    adapter: AdapterConfig = Field(default_factory=AdapterConfig)
    quantization: QuantizationConfig = Field(default_factory=QuantizationConfig)

    @model_validator(mode="after")
    def validate_model_options(self) -> "AnnotatorModelConfig":
        if self.quantization.mode == "4bit" and not self.adapter.enabled:
            raise ValueError("quantization.mode=4bit requires adapter.enabled=true")
        return self


class TrainConfig(BaseModel):
    max_seq_length: int = 1024
    max_steps: int = 10000
    save_steps: int = 500
    learning_rate: float = 5e-4
    per_device_train_batch_size: int = 8
    per_device_eval_batch_size: int = 8
    gradient_accumulation_steps: int = 1
    warmup_steps: int = 0
    logging_steps: int = 50
    bf16: bool = True
    lr_scheduler_type: str = "linear"
    optim: str = "adamw_torch"
    gradient_checkpointing: bool = False
    gpu: GpuConfig = Field(default_factory=GpuConfig)
    deepspeed: DeepSpeedConfig = Field(default_factory=DeepSpeedConfig)

    @model_validator(mode="after")
    def validate_train_options(self) -> "TrainConfig":
        if self.gpu.num_gpus == 1 and self.deepspeed.enabled:
            raise ValueError("train.deepspeed.enabled=false is required for single-GPU training")
        if self.gpu.num_gpus > 1 and not self.deepspeed.enabled:
            raise ValueError("train.deepspeed.enabled=true is required for multi-GPU training")
        return self


class DecodeConfig(BaseModel):
    batch_size: int = 64
    max_new_tokens: int = 1024
    max_length: int = 1024
    min_length: int = 1
    temperature: float = 0.0
    num_beams: int = 1
    length_penalty: float = 1.0
    no_repeat_ngram_size: int = 0
    early_stopping: bool = False
    top_k: int = 50
    top_p: float = 0.9
    do_sample: bool = False
    gpu: GpuConfig = Field(default_factory=GpuConfig)

    @model_validator(mode="after")
    def validate_decode_gpu(self) -> "DecodeConfig":
        if self.gpu.num_gpus != 1:
            raise ValueError("checkpoint selection and annotation currently support only a single GPU")
        return self


class CheckpointSelectionConfig(BaseModel):
    enabled: bool = True
    split: Literal["dev"] = "dev"
    mode: Literal["all", "list"] = "all"
    checkpoints: list[int] = Field(default_factory=list)
    metric: Literal["average_f1", "span_f1", "component_f1", "relation_f1"] = "average_f1"
    decode: DecodeConfig = Field(default_factory=DecodeConfig)

    @model_validator(mode="after")
    def validate_selection(self) -> "CheckpointSelectionConfig":
        if self.mode == "list" and not self.checkpoints:
            raise ValueError("checkpoint_selection.checkpoints is required when mode=list")
        return self


class FinalAnnotationConfig(BaseModel):
    split: Literal["test"] = "test"
    decode: DecodeConfig = Field(default_factory=DecodeConfig)


class PreparedDataConfig(BaseModel):
    output_format: Literal["tanl"] = "tanl"
    fulltext: bool = False


class EvaluationConfig(BaseModel):
    output_format: Literal["tanl"] = "tanl"
    overlap: Optional[float] = None
    space: bool = False


class ParsedComponent(BaseModel):
    type: str
    start: int
    end: int


class ParsedRelation(BaseModel):
    type: str
    source: int
    target: int


class ParsedPrediction(BaseModel):
    doc_id: str
    components: list[ParsedComponent]
    relations: list[ParsedRelation]


class PredictionBatch(BaseModel):
    dataset_name: str
    split: SplitName
    raw_predictions: list[str]
    parsed_predictions: list[ParsedPrediction]


class MetricScore(BaseModel):
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0


class EvaluationMetrics(BaseModel):
    span: MetricScore = Field(default_factory=MetricScore)
    component: MetricScore = Field(default_factory=MetricScore)
    relation: MetricScore = Field(default_factory=MetricScore)
    average_f1: float = 0.0
