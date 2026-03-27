import json
from typing import Optional, Sequence

from am_text2text.schemas.annotator import ParsedPrediction


class TokenRow:
    def __init__(self, *, index: int, token: str, label: str) -> None:
        self.index = index
        self.token = token
        self.label = label

    def to_conll_format(self) -> str:
        return f"{self.index}\t{self.token}\t{self.label}"


class AAECFormatter:
    COMPONENT_LABEL_MAP = {
        "major claim": "MajorClaim",
        "claim": "Claim",
        "claim for": "Claim:For",
        "claim against": "Claim:Against",
        "premise": "Premise",
    }

    RELATION_LABEL_MAP = {
        "support": "Support",
        "attack": "Attack",
    }

    @staticmethod
    def _normalize_key(label: str) -> str:
        return " ".join(label.replace("_", " ").replace("-", " ").split()).lower()

    def format_predictions(
        self,
        predictions: Sequence[ParsedPrediction],
        tokens_list: Sequence[Sequence[str]],
        save_path: str,
    ) -> str:
        with open(save_path, mode="w", encoding="utf-8") as handle:
            for pred, tokens in zip(predictions, tokens_list):
                formatted = self._format_from_parsed(pred, list(tokens))
                handle.write(formatted + "\n\n")
        return save_path

    def _load_gold_tokens(
        self,
        path: str,
        doc_indices: Optional[Sequence[int]] = None,
    ) -> list[list[str]]:
        if path.endswith(".json"):
            with open(path, mode="r", encoding="utf-8") as handle:
                gt_data = json.load(handle)
            tokens_list = []
            for gt in gt_data:
                if "tokens" in gt:
                    tokens_list.append(gt["tokens"])
                elif "input_text" in gt:
                    tokens_list.append(gt["input_text"].split())
                else:
                    raise KeyError("Gold data must contain 'tokens' or 'input_text'")
            return self._select_by_indices(tokens_list, doc_indices, "gold_tokens")

        tokens_list: list[list[str]] = []
        current_tokens: list[str] = []
        with open(path, mode="r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    if current_tokens:
                        tokens_list.append(current_tokens)
                        current_tokens = []
                    continue
                parts = line.split("\t")
                if len(parts) > 1:
                    current_tokens.append(parts[1])
        if current_tokens:
            tokens_list.append(current_tokens)
        return self._select_by_indices(tokens_list, doc_indices, "gold_tokens")

    @staticmethod
    def _select_by_indices(
        items: list[list[str]],
        doc_indices: Optional[Sequence[int]],
        label: str,
    ) -> list[list[str]]:
        if doc_indices is None:
            return items
        if not doc_indices:
            return []
        max_index = max(doc_indices)
        if max_index >= len(items):
            raise ValueError(f"Requested {label} index {max_index} but only {len(items)} docs available.")
        return [items[idx] for idx in doc_indices]

    def _format_from_parsed(self, prediction: ParsedPrediction, tokens: list[str]) -> str:
        rows = [TokenRow(index=i + 1, token=token, label="O") for i, token in enumerate(tokens)]

        for comp in prediction.components:
            label_key = self.COMPONENT_LABEL_MAP.get(self._normalize_key(comp.type))
            if label_key is None:
                continue
            if comp.start < 0 or comp.end <= comp.start or comp.end > len(tokens):
                continue
            rows[comp.start].label = f"B-{label_key}"
            for i in range(comp.start + 1, comp.end):
                rows[i].label = f"I-{label_key}"

        for rel in prediction.relations:
            rel_label = self.RELATION_LABEL_MAP.get(self._normalize_key(rel.type))
            if rel_label is None:
                continue
            if (
                rel.source < 0
                or rel.target < 0
                or rel.source >= len(prediction.components)
                or rel.target >= len(prediction.components)
            ):
                continue
            source_comp = prediction.components[rel.source]
            target_comp = prediction.components[rel.target]
            if source_comp.start < 0 or source_comp.start >= len(tokens):
                continue
            if target_comp.start < 0 or target_comp.end <= target_comp.start or target_comp.end > len(tokens):
                continue

            relation_pointer = source_comp.start + 1
            for i in range(target_comp.start, target_comp.end):
                rows[i].label = f"{rows[i].label}:{relation_pointer}:{rel_label}"

        return "\n".join(row.to_conll_format() for row in rows)
