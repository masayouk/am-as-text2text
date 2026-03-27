import json
import logging
from pathlib import Path

from am_text2text.evaluation.runner import list_checkpoints
from am_text2text.pipeline.annotate import annotate_to_dir, build_reusable_annotation_session
from am_text2text.pipeline.common import ensure_run_dir, payload_matches, snapshot_config, write_stage_status
from am_text2text.pipeline.evaluate import parse_and_evaluate
from am_text2text.schemas.project import AnnotatorProjectConfig

LOGGER = logging.getLogger(__name__)


def run_select_checkpoint_command(config: AnnotatorProjectConfig, config_path: Path) -> dict:
    run_dir = ensure_run_dir(config)
    snapshot_config(config, run_dir)
    selection_path = select_checkpoint(config, run_dir)
    write_stage_status(run_dir, "checkpoint_selected", config_path)
    return {"selected_checkpoint_path": str(selection_path)}


def select_checkpoint(
    config: AnnotatorProjectConfig,
    run_dir: Path,
    reuse_session=None,
) -> Path:
    if not config.checkpoint_selection.enabled:
        raise ValueError("checkpoint_selection.enabled is false")

    model_dir = run_dir / "train" / "model"
    if not model_dir.exists():
        raise FileNotFoundError(f"Training output not found: {model_dir}")

    checkpoints = (
        config.checkpoint_selection.checkpoints
        if config.checkpoint_selection.mode == "list"
        else list_checkpoints(model_dir)
    )
    if not checkpoints:
        done_path = model_dir / "done.json"
        if not done_path.exists():
            raise ValueError("No checkpoints found for checkpoint selection")
        done_payload = json.loads(done_path.read_text(encoding="utf-8"))
        checkpoints = [int(done_payload.get("global_step", 0))]

    selection_root = run_dir / "checkpoint_selection"
    selection_request_path = selection_root / "request.json"
    selected_path = selection_root / "selected_checkpoint.json"
    selection_request = {
        "dataset_name": config.dataset.dataset_name,
        "split": config.checkpoint_selection.split,
        "checkpoints": checkpoints,
        "metric": config.checkpoint_selection.metric,
        "decode": config.checkpoint_selection.decode.model_dump(mode="json"),
        "model_dir": str(model_dir),
    }
    if selected_path.exists() and payload_matches(selection_request_path, selection_request):
        LOGGER.info("Reusing checkpoint selection result: %s", selected_path)
        return selected_path

    owns_reuse_session = False
    if reuse_session is None:
        reuse_session = build_reusable_annotation_session(config, run_dir, checkpoints=checkpoints)
        owns_reuse_session = reuse_session is not None

    best_metric = None
    best_checkpoint = None
    selection_root.mkdir(parents=True, exist_ok=True)
    selection_request_path.write_text(json.dumps(selection_request, ensure_ascii=False, indent=2), encoding="utf-8")
    selection_dir = selection_root / config.checkpoint_selection.split
    selection_dir.mkdir(parents=True, exist_ok=True)

    try:
        for checkpoint in checkpoints:
            checkpoint_dir = model_dir / f"checkpoint-{checkpoint}"
            effective_checkpoint_path = checkpoint_dir if checkpoint_dir.exists() else model_dir
            stage_dir = selection_dir / f"checkpoint-{checkpoint}"
            stage_dir.mkdir(parents=True, exist_ok=True)
            raw_prediction_path = annotate_to_dir(
                config=config,
                checkpoint_path=effective_checkpoint_path,
                split=config.checkpoint_selection.split,
                decode=config.checkpoint_selection.decode,
                output_dir=stage_dir,
                reuse_session=reuse_session,
            )
            batch, metrics = parse_and_evaluate(
                config,
                raw_prediction_path,
                config.checkpoint_selection.split,
                stage_dir / "metrics.json",
            )
            (stage_dir / "parsed_predictions.json").write_text(
                json.dumps([prediction.model_dump(mode="json") for prediction in batch.parsed_predictions], ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            metric_value = _metric_value(metrics.model_dump(mode="json"), config.checkpoint_selection.metric)
            if best_metric is None or metric_value > best_metric:
                best_metric = metric_value
                best_checkpoint = checkpoint
    finally:
        if owns_reuse_session and reuse_session is not None:
            reuse_session.close()

    if best_checkpoint is None:
        raise ValueError("Failed to select a checkpoint")

    selected_path.parent.mkdir(parents=True, exist_ok=True)
    selected_path.write_text(
        json.dumps(
            {
                "checkpoint": best_checkpoint,
                "metric": config.checkpoint_selection.metric,
                "metric_value": best_metric,
                "checkpoint_path": str(
                    (model_dir / f"checkpoint-{best_checkpoint}")
                    if (model_dir / f"checkpoint-{best_checkpoint}").exists()
                    else model_dir
                ),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return selected_path


def _metric_value(metrics_payload: dict, metric_name: str) -> float:
    if metric_name == "average_f1":
        return float(metrics_payload["average_f1"])
    if metric_name == "span_f1":
        return float(metrics_payload["span"]["f1"])
    if metric_name == "component_f1":
        return float(metrics_payload["component"]["f1"])
    if metric_name == "relation_f1":
        return float(metrics_payload["relation"]["f1"])
    raise ValueError(f"Unsupported metric: {metric_name}")
