from pathlib import Path
from typing import Dict, Literal, Optional, Union

from pydantic import BaseModel, Field, field_validator

from am_text2text.schemas.common import SplitName


class TokenOffset(BaseModel):
    start: int
    end: int


class Component(BaseModel):
    component_id: str
    type: str
    start: int
    end: int
    text: str


class Relation(BaseModel):
    relation_id: str
    type: str
    head_component_id: str
    tail_component_id: str


class DatasetAnnotationSchema(BaseModel):
    dataset_names: tuple[str, ...]
    entity_types: frozenset[str]
    relation_types: frozenset[str]
    component_aliases: dict[str, str] = Field(default_factory=dict)
    relation_aliases: dict[str, str] = Field(default_factory=dict)
    source_label_prefixes: tuple[str, ...] = ()

    def normalize_component_label(self, label: str) -> str:
        cleaned = _normalize_annotation_label(label)
        return self.component_aliases.get(cleaned, cleaned.replace(" ", "_"))

    def normalize_relation_label(self, label: str) -> str:
        cleaned = _normalize_annotation_label(label)
        return self.relation_aliases.get(cleaned, cleaned.replace(" ", "_"))

    def normalize_source_component_label(self, label: str) -> str:
        cleaned = _normalize_annotation_label(label, prefixes=self.source_label_prefixes)
        if cleaned == "majorclaim":
            cleaned = "major claim"
        if cleaned == "claim:for":
            cleaned = "claim for"
        if cleaned == "claim:against":
            cleaned = "claim against"
        return self.normalize_component_label(cleaned)

    def normalize_source_relation_label(self, label: str) -> str:
        cleaned = _normalize_annotation_label(label, prefixes=self.source_label_prefixes)
        return self.normalize_relation_label(cleaned)


class SharedAliasAnnotationSchema(DatasetAnnotationSchema):
    component_aliases: dict[str, str] = Field(
        default_factory=lambda: {
            "major claim": "major_claim",
            "claim for": "claim_for",
            "claim against": "claim_against",
            "evidence": "premise",
            "claim": "claim",
            "premise": "premise",
        }
    )


class EssayParagraphAnnotationSchema(SharedAliasAnnotationSchema):
    dataset_names: tuple[str, ...] = ("essay", "paragraph")
    entity_types: frozenset[str] = frozenset({"major claim", "claim", "claim for", "claim against", "premise"})
    relation_types: frozenset[str] = frozenset({"support", "attack"})


class CdcpAnnotationSchema(SharedAliasAnnotationSchema):
    dataset_names: tuple[str, ...] = ("cdcp",)
    entity_types: frozenset[str] = frozenset({"value", "policy", "testimony", "fact", "reference"})
    relation_types: frozenset[str] = frozenset({"reason", "evidence"})
    source_label_prefixes: tuple[str, ...] = ("cdcp_",)


class AbstrctAnnotationSchema(SharedAliasAnnotationSchema):
    dataset_names: tuple[str, ...] = ("abstrct", "abstruct")
    entity_types: frozenset[str] = frozenset({"major claim", "claim", "premise"})
    relation_types: frozenset[str] = frozenset({"support", "attack", "partial attack"})
    source_label_prefixes: tuple[str, ...] = ("abstrct_",)


class DocumentRecord(BaseModel):
    doc_id: str
    dataset_name: str
    split: SplitName
    raw_text: str
    tokens: list[str]
    token_offsets: list[TokenOffset]
    components: list[Component] = Field(default_factory=list)
    relations: list[Relation] = Field(default_factory=list)
    metadata: Dict[str, Optional[Union[str, int, float, bool]]] = Field(default_factory=dict)


class DatasetSplit(BaseModel):
    split: SplitName
    documents: list[DocumentRecord]


class BaseExample(BaseModel):
    example_id: str
    doc_id: str
    split: SplitName
    input_text: str
    output_text: str
    metadata: Dict[str, Optional[Union[str, int, float, bool]]] = Field(default_factory=dict)


class BaseDatasetSplit(BaseModel):
    split: SplitName
    examples: list[BaseExample]


class BaseDatasetBundle(BaseModel):
    dataset_name: str
    task_name: str
    splits: dict[SplitName, BaseDatasetSplit]

    def count_examples(self) -> dict[SplitName, int]:
        return {split_name: len(split.examples) for split_name, split in self.splits.items()}


class CanonicalDataset(BaseModel):
    dataset_name: str
    source_type: Literal["conll", "mrp"]
    source_dir: Path
    splits: dict[SplitName, DatasetSplit]

    def document_counts(self) -> dict[SplitName, int]:
        return {split_name: len(split.documents) for split_name, split in self.splits.items()}


class DatasetSourceConfig(BaseModel):
    dataset_name: str
    source_type: Literal["conll", "mrp"]
    source_dir: Path
    splits: list[SplitName] = Field(default_factory=lambda: ["train", "dev", "test"])
    conll_token_pattern: str = "{split}.dat"
    conll_label_pattern: str = "{split}.dat.abs"
    mrp_pattern: str = "{dataset_name}_{split}.mrp"

    @field_validator("source_dir")
    @classmethod
    def expand_source_dir(cls, value: Path) -> Path:
        return value.expanduser()

    def token_path(self, split: SplitName) -> Path:
        return self.source_dir / self.conll_token_pattern.format(split=split)

    def label_path(self, split: SplitName) -> Path:
        return self.source_dir / self.conll_label_pattern.format(split=split)

    def mrp_path(self, split: SplitName) -> Path:
        return self.source_dir / self.mrp_pattern.format(dataset_name=self.dataset_name, split=split)


def resolve_dataset_annotation_schema(dataset_name: str) -> DatasetAnnotationSchema:
    normalized = dataset_name.lower()
    if normalized in {"essay", "paragraph"}:
        return EssayParagraphAnnotationSchema()
    if normalized == "cdcp":
        return CdcpAnnotationSchema()
    if normalized in {"abstrct", "abstruct"}:
        return AbstrctAnnotationSchema()
    raise ValueError(f"Unsupported dataset annotation schema: {dataset_name}")


def _normalize_annotation_label(label: str, *, prefixes: tuple[str, ...] = ()) -> str:
    cleaned = label.strip().lower()
    for prefix in prefixes:
        cleaned = cleaned.replace(prefix, "")
    return " ".join(cleaned.replace("_", " ").replace("-", " ").split()).lower()
