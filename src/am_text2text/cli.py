import argparse
import json
from pathlib import Path
from typing import Optional, Sequence

from am_text2text.pipeline import (
    build_training_data,
    materialize_dataset,
    run_annotate_command,
    run_evaluate_command,
    run_pipeline,
    run_select_checkpoint_command,
    run_train_command,
)
from am_text2text.project import load_project_config
from am_text2text.utils import setup_logging


def main(argv: Optional[Sequence[str]] = None) -> None:
    setup_logging()

    parser = argparse.ArgumentParser(prog="am-text2text")
    subparsers = parser.add_subparsers(dest="command", required=True)

    for command in [
        "materialize-dataset",
        "prepare-annotator-dataset",
        "train-annotator",
        "select-checkpoint",
        "annotate",
        "evaluate",
        "run",
    ]:
        subparser = subparsers.add_parser(command)
        subparser.add_argument("--config", required=True, help="Path to annotator YAML config")

    args = parser.parse_args(list(argv) if argv is not None else None)
    config = load_project_config(args.config)

    if args.command == "materialize-dataset":
        path = materialize_dataset(config)
        print(json.dumps({"canonical_path": str(path)}, ensure_ascii=False, indent=2))
        return

    if args.command == "prepare-annotator-dataset":
        path = build_training_data(config)
        print(json.dumps({"dataset_dir": str(path)}, ensure_ascii=False, indent=2))
        return

    command_handlers = {
        "train-annotator": run_train_command,
        "select-checkpoint": run_select_checkpoint_command,
        "annotate": run_annotate_command,
        "evaluate": run_evaluate_command,
        "run": run_pipeline,
    }
    result = command_handlers[args.command](config, Path(args.config).resolve())
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
