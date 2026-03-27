import json
from datetime import datetime, timezone
from pathlib import Path

import yaml

from am_text2text.schemas.annotator import EvaluationMetrics
from am_text2text.schemas.project import AnnotatorProjectConfig


def ensure_run_dir(config: AnnotatorProjectConfig) -> Path:
    run_dir = config.run_dir()
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def snapshot_config(config: AnnotatorProjectConfig, run_dir: Path) -> None:
    payload = config.model_dump(mode="json")
    (run_dir / "config.yaml").write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def write_run_metadata(run_dir: Path, payload: dict) -> None:
    (run_dir / "run.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_stage_status(run_dir: Path, status: str, config_path: Path) -> None:
    write_run_metadata(
        run_dir,
        {
            "status": status,
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "config_path": str(config_path),
        },
    )


def write_metrics(path: Path, metrics: EvaluationMetrics) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(metrics.model_dump_json(indent=2), encoding="utf-8")


def latest_checkpoint_dir(model_dir: Path, checkpoints: list[int]) -> Path | None:
    if not checkpoints:
        return None
    return model_dir / f"checkpoint-{max(checkpoints)}"


def selected_checkpoint_path(run_dir: Path) -> Path:
    selected_path = run_dir / "checkpoint_selection" / "selected_checkpoint.json"
    if not selected_path.exists():
        raise FileNotFoundError(f"Selected checkpoint file not found: {selected_path}")
    payload = json.loads(selected_path.read_text(encoding="utf-8"))
    return Path(payload["checkpoint_path"])


def payload_matches(path: Path, payload: dict, *, ignored_keys: set[str] | None = None) -> bool:
    if not path.exists():
        return False
    stored_payload = json.loads(path.read_text(encoding="utf-8"))
    effective_ignored_keys = ignored_keys or set()
    return _normalized_payload(stored_payload, effective_ignored_keys) == _normalized_payload(
        payload,
        effective_ignored_keys,
    )


def clear_annotation_outputs(output_dir: Path) -> None:
    for path in (
        output_dir / "raw_predictions.json",
        output_dir / "parsed_predictions.json",
        output_dir / "metrics.json",
    ):
        if path.exists():
            path.unlink()


def _normalized_payload(value, ignored_keys: set[str]):
    if isinstance(value, dict):
        return {
            key: _normalized_payload(item, ignored_keys)
            for key, item in sorted(value.items())
            if key not in ignored_keys
        }
    if isinstance(value, list):
        return [_normalized_payload(item, ignored_keys) for item in value]
    return value
