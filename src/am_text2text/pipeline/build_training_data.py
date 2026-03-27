import json
import shutil
from pathlib import Path

from am_text2text.data import build_base_dataset, write_base_dataset_bundle
from am_text2text.pipeline.import_dataset import load_canonical_dataset
from am_text2text.schemas.project import AnnotatorProjectConfig


def build_training_data(config: AnnotatorProjectConfig) -> Path:
    dataset = load_canonical_dataset(config)
    bundle = build_base_dataset(
        dataset,
        fulltext=config.prepared_data.fulltext,
        task_name="annotator_tanl",
    )
    output_root = config.annotator_dataset_root_dir()
    dataset_dir = config.annotator_dataset_output_dir()
    if dataset_dir.exists():
        shutil.rmtree(dataset_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    dataset_dir = write_base_dataset_bundle(bundle, output_root)
    manifest_path = dataset_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "dataset_name": bundle.dataset_name,
                "task_name": bundle.task_name,
                "split_example_counts": bundle.count_examples(),
                "output_format": config.prepared_data.output_format,
                "fulltext": config.prepared_data.fulltext,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return dataset_dir
