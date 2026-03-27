import json
from pathlib import Path

from am_text2text.evaluation.runner import evaluate_prediction_batch, parse_prediction_batch
from am_text2text.pipeline.common import ensure_run_dir, snapshot_config, write_metrics, write_stage_status
from am_text2text.pipeline.import_dataset import load_canonical_dataset
from am_text2text.schemas.project import AnnotatorProjectConfig


def run_evaluate_command(config: AnnotatorProjectConfig, config_path: Path) -> dict:
    run_dir = ensure_run_dir(config)
    snapshot_config(config, run_dir)
    metrics_path = evaluate_split(config, run_dir, split=config.annotation.split, decode_stage="test")
    write_stage_status(run_dir, "evaluated", config_path)
    return {"metrics_path": str(metrics_path)}


def evaluate_split(config: AnnotatorProjectConfig, run_dir: Path, *, split: str, decode_stage: str) -> Path:
    raw_prediction_path = run_dir / decode_stage / "raw_predictions.json"
    if not raw_prediction_path.exists():
        raise FileNotFoundError(f"Prediction file not found: {raw_prediction_path}")
    metrics_path = run_dir / decode_stage / "metrics.json"
    batch, _ = parse_and_evaluate(config, raw_prediction_path, split, metrics_path)
    (run_dir / decode_stage / "parsed_predictions.json").write_text(
        json.dumps([prediction.model_dump(mode="json") for prediction in batch.parsed_predictions], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return metrics_path


def parse_and_evaluate(config: AnnotatorProjectConfig, raw_prediction_path: Path, split: str, metrics_path: Path):
    dataset = load_canonical_dataset(config)
    batch = parse_prediction_batch(
        dataset=dataset,
        split=split,
        raw_prediction_path=raw_prediction_path,
        output_format=config.evaluation.output_format,
    )
    metrics = evaluate_prediction_batch(
        batch=batch,
        source=config.dataset,
        split=split,
        result_path=metrics_path,
        config=config.evaluation,
        canonical_documents=dataset.splits[split].documents,
    )
    write_metrics(metrics_path, metrics)
    return batch, metrics
