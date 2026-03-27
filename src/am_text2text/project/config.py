from pathlib import Path
from typing import Union

import yaml

from am_text2text.schemas.project import AnnotatorProjectConfig


def load_project_config(config_path: Union[str, Path]) -> AnnotatorProjectConfig:
    path = Path(config_path).resolve()
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    config = AnnotatorProjectConfig.model_validate(raw)
    _resolve_paths(config, path.parent)
    return config


def _resolve_paths(config: AnnotatorProjectConfig, base_dir: Path) -> None:
    config.dataset.source_dir = _resolve_path(config.dataset.source_dir, base_dir)
    config.outputs.root_dir = _resolve_path(config.outputs.root_dir, base_dir)
    config.model.prompt.prompt_dir = _resolve_path(config.model.prompt.prompt_dir, base_dir)
    if config.train.deepspeed.config_path is not None:
        config.train.deepspeed.config_path = _resolve_path(config.train.deepspeed.config_path, base_dir)


def _resolve_path(path: Path, base_dir: Path) -> Path:
    if path.is_absolute():
        return path
    return (base_dir / path).resolve()
