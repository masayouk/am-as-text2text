import json
import re
from pathlib import Path
from typing import Any, Optional, Tuple

from am_text2text.schemas.dataset import (
    CanonicalDataset,
    Component,
    DatasetAnnotationSchema,
    DatasetSourceConfig,
    DatasetSplit,
    DocumentRecord,
    Relation,
    TokenOffset,
    resolve_dataset_annotation_schema,
)


def load_conll_dataset(source: DatasetSourceConfig) -> CanonicalDataset:
    splits: dict[str, DatasetSplit] = {}

    for split in source.splits:
        token_docs = _read_conll_docs(source.token_path(split))
        label_docs = _read_conll_docs(source.label_path(split))
        if len(token_docs) != len(label_docs):
            raise ValueError(
                f"Document count mismatch for split={split}: "
                f"tokens={len(token_docs)} labels={len(label_docs)}"
            )

        documents: list[DocumentRecord] = []
        for doc_index, (token_doc, label_doc) in enumerate(zip(token_docs, label_docs)):
            tokens = _extract_tokens(token_doc)
            labels = _extract_labels(label_doc)
            label_tokens = _extract_tokens(label_doc)

            if tokens != label_tokens:
                raise ValueError(
                    f"Token mismatch for split={split} doc={doc_index}: "
                    f"{source.token_path(split)} vs {source.label_path(split)}"
                )

            components, relations = _labels_to_components_and_relations(tokens, labels)
            raw_text, token_offsets = _tokens_to_text_and_offsets(tokens)
            documents.append(
                DocumentRecord(
                    doc_id=f"{source.dataset_name}_{split}_{doc_index}",
                    dataset_name=source.dataset_name,
                    split=split,
                    raw_text=raw_text,
                    tokens=tokens,
                    token_offsets=token_offsets,
                    components=components,
                    relations=relations,
                    metadata={
                        "source_type": "conll",
                        "source_tokens_path": str(source.token_path(split)),
                        "source_labels_path": str(source.label_path(split)),
                    },
                )
            )

        splits[split] = DatasetSplit(split=split, documents=documents)

    return CanonicalDataset(
        dataset_name=source.dataset_name,
        source_type=source.source_type,
        source_dir=source.source_dir,
        splits=splits,
    )


def load_mrp_dataset(source: DatasetSourceConfig) -> CanonicalDataset:
    splits: dict[str, DatasetSplit] = {}
    annotation_schema = resolve_dataset_annotation_schema(source.dataset_name)

    for split in source.splits:
        path = source.mrp_path(split)
        if not path.exists():
            raise FileNotFoundError(f"MRP file not found: {path}")

        documents: list[DocumentRecord] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            raw = line.strip()
            if not raw:
                continue
            record = json.loads(raw)
            documents.append(
                _record_to_document(
                    record=record,
                    dataset_name=source.dataset_name,
                    split=split,
                    annotation_schema=annotation_schema,
                )
            )

        splits[split] = DatasetSplit(split=split, documents=documents)

    return CanonicalDataset(
        dataset_name=source.dataset_name,
        source_type=source.source_type,
        source_dir=source.source_dir,
        splits=splits,
    )


def _read_conll_docs(path: Path) -> list[list[list[str]]]:
    if not path.exists():
        raise FileNotFoundError(f"CONLL file not found: {path}")

    docs: list[list[list[str]]] = []
    current: list[list[str]] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            if current:
                docs.append(current)
                current = []
            continue
        current.append(line.split("\t"))
    if current:
        docs.append(current)
    return docs


def _extract_tokens(doc: list[list[str]]) -> list[str]:
    return [row[1] for row in doc if len(row) > 1]


def _extract_labels(doc: list[list[str]]) -> list[str]:
    return [row[-1] for row in doc]


def _tokens_to_text_and_offsets(tokens: list[str]) -> tuple[str, list[TokenOffset]]:
    offsets: list[TokenOffset] = []
    cursor = 0
    for token in tokens:
        start = cursor
        end = start + len(token)
        offsets.append(TokenOffset(start=start, end=end))
        cursor = end + 1
    return " ".join(tokens), offsets


def _normalize_component_label(label: str) -> str:
    cleaned = label.strip().lower().replace("_", " ").replace("-", " ")
    cleaned = " ".join(cleaned.split())
    if cleaned in {"majorclaim", "major claim"}:
        return "major_claim"
    if cleaned in {"claim for", "claim:for"}:
        return "claim_for"
    if cleaned in {"claim against", "claim:against"}:
        return "claim_against"
    if cleaned == "premise":
        return "premise"
    if cleaned == "claim":
        return "claim"
    return cleaned.replace(" ", "_")


def _normalize_relation_label(label: str) -> str:
    return " ".join(label.strip().lower().replace("_", " ").replace("-", " ").split()).replace(" ", "_")


def _parse_label(raw_label: str) -> tuple[str, Optional[int], Optional[str]]:
    label = raw_label[2:] if raw_label.startswith(("B-", "I-")) else raw_label
    parts = label.split(":")
    base = parts[0]
    if base == "MajorClaim":
        return "major_claim", None, None
    if base == "Claim":
        if len(parts) > 1 and parts[1] in {"For", "Against"}:
            return _normalize_component_label(f"claim {parts[1].lower()}"), None, None
        return "claim", None, None
    if base == "Premise":
        pointer = int(parts[1]) - 1 if len(parts) > 2 and parts[1].isdigit() else None
        relation = _normalize_relation_label(parts[2]) if len(parts) > 2 else None
        return "premise", pointer, relation
    return _normalize_component_label(base), None, None


def _labels_to_components_and_relations(
    tokens: list[str], labels: list[str]
) -> tuple[list[Component], list[Relation]]:
    components: list[Component] = []
    pending_relations: list[tuple[str, Optional[int], Optional[str]]] = []

    current_type: Optional[str] = None
    current_start: Optional[int] = None
    current_pointer: Optional[int] = None
    current_relation: Optional[str] = None

    def flush(end_index: int) -> None:
        nonlocal current_type, current_start, current_pointer, current_relation
        if current_type is None or current_start is None:
            return
        component_id = f"c{len(components)}"
        components.append(
            Component(
                component_id=component_id,
                type=current_type,
                start=current_start,
                end=end_index,
                text=" ".join(tokens[current_start:end_index]),
            )
        )
        pending_relations.append((component_id, current_pointer, current_relation))
        current_type = None
        current_start = None
        current_pointer = None
        current_relation = None

    for token_index, label in enumerate(labels):
        if label == "O":
            flush(token_index)
            continue

        component_type, pointer, relation_type = _parse_label(label)
        starts_new = label.startswith("B-") or current_type is None or current_type != component_type
        if starts_new:
            flush(token_index)
            current_type = component_type
            current_start = token_index
            current_pointer = pointer
            current_relation = relation_type
        else:
            if current_pointer is None:
                current_pointer = pointer
            if current_relation is None:
                current_relation = relation_type

    flush(len(labels))

    component_start_index = {component.start: component.component_id for component in components}
    relations: list[Relation] = []
    for tail_component_id, head_start_index, relation_type in pending_relations:
        if head_start_index is None or relation_type is None:
            continue
        head_component_id = component_start_index.get(head_start_index)
        if head_component_id is None:
            continue
        relations.append(
            Relation(
                relation_id=f"r{len(relations)}",
                type=relation_type,
                head_component_id=head_component_id,
                tail_component_id=tail_component_id,
            )
        )

    return components, relations


def _record_to_document(
    record: dict[str, Any],
    dataset_name: str,
    split: str,
    annotation_schema: DatasetAnnotationSchema,
) -> DocumentRecord:
    raw_text = str(record.get("input", ""))
    tokens, offsets = _tokenize_with_offsets(raw_text)

    node_span_rows: list[tuple[int, str, int, int]] = []
    for node in record.get("nodes", []):
        span = _anchor_span(node.get("anchors", []))
        if span is None:
            continue
        token_span = _char_span_to_token_span(span[0], span[1], offsets)
        if token_span is None:
            continue
        token_start, token_end = token_span
        node_span_rows.append(
            (
                int(node["id"]),
                annotation_schema.normalize_source_component_label(str(node["label"])),
                token_start,
                token_end,
            )
        )

    node_span_rows.sort(key=lambda row: (row[2], row[3], row[0]))

    components: list[Component] = []
    node_id_to_component_id: dict[int, str] = {}
    for node_id, component_type, start, end in node_span_rows:
        component_id = f"c{len(components)}"
        node_id_to_component_id[node_id] = component_id
        components.append(
            Component(
                component_id=component_id,
                type=component_type,
                start=start,
                end=end,
                text=" ".join(tokens[start:end]),
            )
        )

    relations: list[Relation] = []
    for edge in record.get("edges", []):
        head_component_id = node_id_to_component_id.get(int(edge["source"]))
        tail_component_id = node_id_to_component_id.get(int(edge["target"]))
        if head_component_id is None or tail_component_id is None:
            continue
        relations.append(
            Relation(
                relation_id=f"r{len(relations)}",
                type=annotation_schema.normalize_source_relation_label(str(edge["label"])),
                head_component_id=head_component_id,
                tail_component_id=tail_component_id,
            )
        )

    return DocumentRecord(
        doc_id=str(record.get("id")),
        dataset_name=dataset_name,
        split=split,
        raw_text=raw_text,
        tokens=tokens,
        token_offsets=offsets,
        components=components,
        relations=relations,
        metadata={
            "source_type": "mrp",
            "framework": str(record.get("framework", "")),
        },
    )


def _tokenize_with_offsets(text: str) -> tuple[list[str], list[TokenOffset]]:
    tokens: list[str] = []
    offsets: list[TokenOffset] = []
    for match in re.finditer(r"\S+", text):
        tokens.append(match.group(0))
        offsets.append(TokenOffset(start=match.start(), end=match.end()))
    return tokens, offsets


def _anchor_span(anchors: list[dict[str, Any]]) -> Optional[Tuple[int, int]]:
    starts: list[int] = []
    ends: list[int] = []
    for anchor in anchors:
        start = anchor.get("from", anchor.get("start"))
        end = anchor.get("to", anchor.get("end"))
        if start is None or end is None:
            continue
        starts.append(int(start))
        ends.append(int(end))
    if not starts:
        return None
    return min(starts), max(ends)


def _char_span_to_token_span(
    char_start: int, char_end: int, offsets: list[TokenOffset]
) -> Optional[Tuple[int, int]]:
    overlapping = [
        index
        for index, offset in enumerate(offsets)
        if offset.end > char_start and offset.start < char_end
    ]
    if not overlapping:
        return None
    return overlapping[0], overlapping[-1] + 1
