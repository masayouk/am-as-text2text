import json
from collections import defaultdict
from pathlib import Path

from am_text2text.schemas.dataset import BaseDatasetBundle, BaseDatasetSplit, BaseExample, CanonicalDataset


def build_base_dataset(
    dataset: CanonicalDataset,
    *,
    fulltext: bool = False,
    task_name: str = "annotator_tanl",
) -> BaseDatasetBundle:
    splits: dict[str, BaseDatasetSplit] = {}
    for split_name, split in dataset.splits.items():
        examples: list[BaseExample] = []
        for doc in split.documents:
            examples.append(
                BaseExample(
                    example_id=doc.doc_id,
                    doc_id=doc.doc_id,
                    split=split_name,
                    input_text=" ".join(doc.tokens),
                    output_text=_build_output_text(doc, fulltext=fulltext),
                    metadata={"fulltext": fulltext},
                )
            )
        splits[split_name] = BaseDatasetSplit(split=split_name, examples=examples)
    return BaseDatasetBundle(dataset_name=dataset.dataset_name, task_name=task_name, splits=splits)


def write_base_dataset_bundle(bundle: BaseDatasetBundle, root_dir: Path) -> Path:
    dataset_dir = root_dir / bundle.dataset_name
    dataset_dir.mkdir(parents=True, exist_ok=True)
    for split_name, split in bundle.splits.items():
        output_path = dataset_dir / f"{bundle.dataset_name}_{split_name}.json"
        rows = [example.model_dump(mode="json") for example in split.examples]
        output_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    return dataset_dir


def _build_output_text(doc, *, fulltext: bool) -> str:
    span_texts = {
        component.component_id: " ".join(doc.tokens[component.start : component.end])
        for component in doc.components
    }
    relations_by_tail: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for relation in doc.relations:
        if relation.tail_component_id in span_texts and relation.head_component_id in span_texts:
            relations_by_tail[relation.tail_component_id].append(
                (
                    relation.type.lower().replace("_", " "),
                    span_texts[relation.head_component_id],
                )
            )

    ordered_components = sorted(doc.components, key=lambda comp: (comp.start, comp.end))
    annotations: list[str] = []
    cursor = 0
    for component in ordered_components:
        if fulltext and cursor < component.start:
            annotations.append(" ".join(doc.tokens[cursor : component.start]))
        label = _format_component_label(component.type)
        if component.component_id in relations_by_tail:
            relation_labels = [
                f"{relation_type} = {head_span}"
                for relation_type, head_span in relations_by_tail[component.component_id]
            ]
            label = " | ".join([label, *relation_labels])
        annotations.append(f"[ {span_texts[component.component_id]} | {label} ]")
        cursor = component.end

    if fulltext and cursor < len(doc.tokens):
        annotations.append(" ".join(doc.tokens[cursor:]))

    return " ".join(part for part in annotations if part).strip()


def _format_component_label(label: str) -> str:
    cleaned = label.lower().replace("_", " ").strip()
    if cleaned == "major claim":
        return "major claim"
    return cleaned
