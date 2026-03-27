from pathlib import Path

from am_text2text.inference.runner import (
    AnnotationRequest,
    ReusableAnnotationSession,
    run_annotation_request,
)
from am_text2text.pipeline.build_training_data import build_training_data
from am_text2text.pipeline.common import (
    clear_annotation_outputs,
    ensure_run_dir,
    payload_matches,
    selected_checkpoint_path,
    snapshot_config,
    write_stage_status,
)
from am_text2text.schemas.project import AnnotatorProjectConfig


def run_annotate_command(config: AnnotatorProjectConfig, config_path: Path) -> dict:
    run_dir = ensure_run_dir(config)
    snapshot_config(config, run_dir)
    raw_prediction_path = annotate_split(config, run_dir, split=config.annotation.split, decode_stage="test")
    write_stage_status(run_dir, "annotated", config_path)
    return {"raw_prediction_path": str(raw_prediction_path)}


def annotate_split(
    config: AnnotatorProjectConfig,
    run_dir: Path,
    *,
    split: str,
    decode_stage: str,
    reuse_session: ReusableAnnotationSession | None = None,
) -> Path:
    checkpoint_path = selected_checkpoint_path(run_dir)
    stage_dir = run_dir / decode_stage
    stage_dir.mkdir(parents=True, exist_ok=True)
    decode = config.annotation.decode if decode_stage == "test" else config.checkpoint_selection.decode
    return annotate_to_dir(
        config=config,
        checkpoint_path=checkpoint_path,
        split=split,
        decode=decode,
        output_dir=stage_dir,
        reuse_session=reuse_session,
    )


def build_reusable_annotation_session(
    config: AnnotatorProjectConfig,
    run_dir: Path,
    *,
    checkpoints: list[int] | None = None,
) -> ReusableAnnotationSession | None:
    model_dir = run_dir / "train" / "model"
    if checkpoints is None:
        from am_text2text.evaluation.runner import list_checkpoints

        checkpoints = list_checkpoints(model_dir)
        if not checkpoints:
            return None
    first_checkpoint = checkpoints[0]
    checkpoint_dir = model_dir / f"checkpoint-{first_checkpoint}"
    effective_checkpoint_path = checkpoint_dir if checkpoint_dir.exists() else model_dir
    if not (effective_checkpoint_path / "adapter_config.json").exists():
        return None

    request = AnnotationRequest(
        data_dir=config.annotator_dataset_output_dir().parent,
        dataset_name=config.dataset.dataset_name,
        checkpoint_path=effective_checkpoint_path,
        output_dir=run_dir / "checkpoint_selection",
        split=config.checkpoint_selection.split,
        model=config.model,
        max_seq_length=config.train.max_seq_length,
        decode=config.checkpoint_selection.decode,
        seed=config.run.seed,
    )
    return ReusableAnnotationSession(request)


def annotate_to_dir(
    *,
    config: AnnotatorProjectConfig,
    checkpoint_path: Path,
    split: str,
    decode,
    output_dir: Path,
    reuse_session: ReusableAnnotationSession | None = None,
) -> Path:
    dataset_dir = config.annotator_dataset_output_dir()
    if not dataset_dir.exists():
        build_training_data(config)
    request = AnnotationRequest(
        data_dir=dataset_dir.parent,
        dataset_name=config.dataset.dataset_name,
        checkpoint_path=checkpoint_path,
        output_dir=output_dir,
        split=split,
        model=config.model,
        max_seq_length=config.train.max_seq_length,
        decode=decode,
        seed=config.run.seed,
    )
    request_path = output_dir / "request.json"
    raw_prediction_path = output_dir / "raw_predictions.json"
    request_payload = request.model_dump(mode="json")
    if raw_prediction_path.exists() and payload_matches(request_path, request_payload):
        return raw_prediction_path

    clear_annotation_outputs(output_dir)
    request_path.write_text(request.model_dump_json(indent=2), encoding="utf-8")
    return run_annotation_request(request, session=reuse_session)
