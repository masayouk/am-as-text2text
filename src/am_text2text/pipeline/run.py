from pathlib import Path

from am_text2text.pipeline.annotate import annotate_split, build_reusable_annotation_session
from am_text2text.pipeline.build_training_data import build_training_data
from am_text2text.pipeline.common import ensure_run_dir, snapshot_config, write_stage_status
from am_text2text.pipeline.evaluate import evaluate_split
from am_text2text.pipeline.import_dataset import materialize_dataset
from am_text2text.pipeline.select_checkpoint import select_checkpoint
from am_text2text.pipeline.train import train_annotator
from am_text2text.schemas.project import AnnotatorProjectConfig


def run_pipeline(config: AnnotatorProjectConfig, config_path: Path) -> dict:
    run_dir = ensure_run_dir(config)
    snapshot_config(config, run_dir)

    materialize_dataset(config)
    build_training_data(config)
    train_annotator(config, run_dir)

    reuse_session = build_reusable_annotation_session(config, run_dir)
    try:
        select_checkpoint(config, run_dir, reuse_session=reuse_session)
        annotate_split(
            config,
            run_dir,
            split=config.annotation.split,
            decode_stage="test",
            reuse_session=reuse_session,
        )
    finally:
        if reuse_session is not None:
            reuse_session.close()

    metrics_path = evaluate_split(config, run_dir, split=config.annotation.split, decode_stage="test")
    write_stage_status(run_dir, "completed", config_path)
    return {"run_dir": str(run_dir), "metrics_path": str(metrics_path)}
