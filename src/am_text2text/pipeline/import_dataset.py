import json
from pathlib import Path

from am_text2text.data import load_conll_dataset, load_mrp_dataset
from am_text2text.schemas.dataset import CanonicalDataset, DatasetSourceConfig
from am_text2text.schemas.project import AnnotatorProjectConfig


def build_canonical_dataset(source: DatasetSourceConfig) -> CanonicalDataset:
    if source.source_type == "conll":
        return load_conll_dataset(source)
    if source.source_type == "mrp":
        return load_mrp_dataset(source)
    raise ValueError(f"Unsupported source_type: {source.source_type}")


def materialize_dataset(config: AnnotatorProjectConfig) -> Path:
    output_dir = config.canonical_output_dir()
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset = build_canonical_dataset(config.dataset)
    canonical_path = output_dir / "canonical.json"
    manifest_path = output_dir / "manifest.json"
    canonical_path.write_text(dataset.model_dump_json(indent=2), encoding="utf-8")
    manifest_path.write_text(
        json.dumps(
            {
                "dataset_name": dataset.dataset_name,
                "source_type": dataset.source_type,
                "source_dir": str(dataset.source_dir),
                "document_counts": dataset.document_counts(),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return canonical_path


def load_canonical_dataset(config: AnnotatorProjectConfig) -> CanonicalDataset:
    canonical_path = config.canonical_output_dir() / "canonical.json"
    if not canonical_path.exists():
        materialize_dataset(config)
    return CanonicalDataset.model_validate_json(canonical_path.read_text(encoding="utf-8"))
