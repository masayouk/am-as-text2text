from pathlib import Path

from pydantic import BaseModel

from am_text2text.schemas.annotator import (
    AnnotatorModelConfig,
    CheckpointSelectionConfig,
    EvaluationConfig,
    FinalAnnotationConfig,
    PreparedDataConfig,
    TrainConfig,
)
from am_text2text.schemas.common import OutputConfig, RunConfig
from am_text2text.schemas.dataset import DatasetSourceConfig


class AnnotatorProjectConfig(BaseModel):
    run: RunConfig
    dataset: DatasetSourceConfig
    prepared_data: PreparedDataConfig = PreparedDataConfig()
    model: AnnotatorModelConfig
    train: TrainConfig = TrainConfig()
    checkpoint_selection: CheckpointSelectionConfig = CheckpointSelectionConfig()
    annotation: FinalAnnotationConfig = FinalAnnotationConfig()
    evaluation: EvaluationConfig = EvaluationConfig()
    outputs: OutputConfig = OutputConfig()

    def canonical_output_dir(self) -> Path:
        return self.outputs.root_dir / "prepared" / "canonical" / self.dataset.dataset_name

    def annotator_dataset_root_dir(self) -> Path:
        return self.outputs.root_dir / "prepared" / "annotator_dataset"

    def annotator_dataset_output_dir(self) -> Path:
        return self.annotator_dataset_root_dir() / self.dataset.dataset_name

    def run_dir(self) -> Path:
        return self.outputs.root_dir / "runs" / self.run.name
