import logging
import shutil
from pathlib import Path

from am_text2text.evaluation.runner import list_checkpoints
from am_text2text.pipeline.build_training_data import build_training_data
from am_text2text.pipeline.common import (
    ensure_run_dir,
    latest_checkpoint_dir,
    payload_matches,
    snapshot_config,
    write_stage_status,
)
from am_text2text.schemas.project import AnnotatorProjectConfig
from am_text2text.training.launcher import run_training_worker
from am_text2text.training.worker import TrainWorkerRequest, run_train_request

LOGGER = logging.getLogger(__name__)


def run_train_command(config: AnnotatorProjectConfig, config_path: Path) -> dict:
    run_dir = ensure_run_dir(config)
    snapshot_config(config, run_dir)
    train_dir = train_annotator(config, run_dir)
    write_stage_status(run_dir, "trained", config_path)
    return {"train_dir": str(train_dir)}


def train_annotator(config: AnnotatorProjectConfig, run_dir: Path) -> Path:
    dataset_dir = build_training_data(config)
    train_dir = run_dir / "train"
    model_dir = train_dir / "model"
    request_path = train_dir / "request.json"
    train_dir.mkdir(parents=True, exist_ok=True)

    request = TrainWorkerRequest(
        data_dir=dataset_dir.parent,
        dataset_name=config.dataset.dataset_name,
        output_dir=model_dir,
        model=config.model,
        train=config.train,
        seed=config.run.seed,
    )
    if _is_completed_train(request, request_path, model_dir):
        LOGGER.info("Reusing completed training output: %s", model_dir)
        return train_dir

    resume_from_checkpoint = None
    if _can_resume_train(request, request_path, model_dir):
        resume_from_checkpoint = latest_checkpoint_dir(model_dir, list_checkpoints(model_dir))
        LOGGER.info("Resuming training from checkpoint: %s", resume_from_checkpoint)
    else:
        if model_dir.exists():
            shutil.rmtree(model_dir)
        model_dir.mkdir(parents=True, exist_ok=True)

    effective_request = request.model_copy(update={"resume_from_checkpoint": resume_from_checkpoint})
    if config.train.gpu.num_gpus > 1:
        run_training_worker(
            "am_text2text.training.worker",
            effective_request,
            request_path,
            gpu=config.train.gpu,
        )
    else:
        request_path.write_text(effective_request.model_dump_json(indent=2), encoding="utf-8")
        run_train_request(effective_request)

    return train_dir


def _is_completed_train(request: TrainWorkerRequest, request_path: Path, model_dir: Path) -> bool:
    return (model_dir / "done.json").exists() and payload_matches(
        request_path,
        request.model_dump(mode="json"),
        ignored_keys={"resume_from_checkpoint"},
    )


def _can_resume_train(request: TrainWorkerRequest, request_path: Path, model_dir: Path) -> bool:
    if not model_dir.exists():
        return False
    checkpoints = list_checkpoints(model_dir)
    return latest_checkpoint_dir(model_dir, checkpoints) is not None and payload_matches(
        request_path,
        request.model_dump(mode="json"),
        ignored_keys={"resume_from_checkpoint"},
    )
