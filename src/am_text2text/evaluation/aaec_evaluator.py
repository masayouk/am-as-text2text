import json
import os
import shutil
import tempfile
from contextlib import contextmanager
from typing import Optional

from am_text2text.evaluation.upstream.aaec_acl2017 import (
    compute_metrics_component,
    compute_metrics_relation,
    compute_metrics_span,
)
from am_text2text.evaluation.upstream.aaec_acl2017.aaec_formatter import AAECFormatter
from am_text2text.evaluation.upstream.aaec_acl2017.docReader import readDocs
from am_text2text.evaluation.evaluator_base import BaseEvaluator
from am_text2text.schemas.annotator import EvaluationMetrics, MetricScore, ParsedPrediction


def _doc_id_to_index(doc_id: str) -> Optional[int]:
    parts = str(doc_id).rsplit("_", 1)
    if len(parts) != 2 or not parts[1].isdigit():
        return None
    return int(parts[1])


def _doc_ids_to_indices(doc_ids: list[str]) -> list[int]:
    indices: list[int] = []
    missing: list[str] = []
    for doc_id in doc_ids:
        index = _doc_id_to_index(str(doc_id))
        if index is None:
            missing.append(str(doc_id))
        else:
            indices.append(index)
    if missing:
        preview = ", ".join(missing[:5])
        raise ValueError(f"Unable to extract doc index from doc_ids: {preview}")
    return indices


def _filter_conll_docs_by_indices(path: str, doc_indices: list[int]):
    if not doc_indices:
        return []
    docs = readDocs(path)
    max_index = max(doc_indices)
    if max_index >= len(docs):
        raise ValueError(f"Requested conll index {max_index} but only {len(docs)} docs available in {path}.")
    return [docs[idx] for idx in doc_indices]


def _write_conll_docs(path: str, docs) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        for doc in docs:
            for line in doc:
                handle.write(f"{line}\n")
            handle.write("\n")


class AAECEvaluator(BaseEvaluator):
    def parse_predictions(self, prediction_path: str, gold_path: str, **kwargs):
        del kwargs
        with open(prediction_path, "r", encoding="utf-8") as handle:
            pred_data = json.load(handle)
        preds_list = pred_data.get("predictions", pred_data) if isinstance(pred_data, dict) else pred_data
        if not isinstance(preds_list, list):
            preds_list = [preds_list]
        parsed_predictions = [ParsedPrediction.model_validate(pred) for pred in preds_list]
        return parsed_predictions, gold_path

    def evaluate_metrics(self, parsed_predictions, gold_data, **kwargs):
        labels_path = kwargs.get("labels_path")
        doc_ids = kwargs.get("doc_ids")
        with self._temp_directory() as temp_dir:
            formatter = AAECFormatter()
            doc_indices = _doc_ids_to_indices(doc_ids) if doc_ids is not None else None
            formatted_pred_path = os.path.join(temp_dir, "temp_prediction")
            gold_tokens = formatter._load_gold_tokens(gold_data, doc_indices=doc_indices)
            formatter.format_predictions(parsed_predictions, gold_tokens, formatted_pred_path)

            if not labels_path:
                raise FileNotFoundError("labels_path is required for AAEC evaluation")
            if not os.path.exists(labels_path):
                raise FileNotFoundError(f"Gold org file not found: {labels_path}")

            if doc_indices is not None:
                filtered_gold_path = os.path.join(temp_dir, "filtered_gold.abs")
                filtered_docs = _filter_conll_docs_by_indices(labels_path, doc_indices)
                _write_conll_docs(filtered_gold_path, filtered_docs)
                gold_eval_path = filtered_gold_path
            else:
                gold_eval_path = labels_path

            s_precision, s_recall, s_f1 = compute_metrics_span(formatted_pred_path, gold_eval_path)
            c_precision, c_recall, c_f1 = compute_metrics_component(formatted_pred_path, gold_eval_path)
            r_precision, r_recall, r_f1 = compute_metrics_relation(formatted_pred_path, gold_eval_path)
            return EvaluationMetrics(
                span=MetricScore(precision=s_precision, recall=s_recall, f1=s_f1),
                component=MetricScore(precision=c_precision, recall=c_recall, f1=c_f1),
                relation=MetricScore(precision=r_precision, recall=r_recall, f1=r_f1),
                average_f1=(r_f1 + c_f1) / 2,
            )

    @staticmethod
    @contextmanager
    def _temp_directory():
        temp_dir = tempfile.mkdtemp(prefix="aaec_eval_")
        try:
            yield temp_dir
        finally:
            shutil.rmtree(temp_dir)

    def _get_gold_path(self, split: str, dataset_name: str, base_data_dir: Optional[str] = None, **kwargs) -> str:
        del split, dataset_name, base_data_dir
        token_path = kwargs.get("token_path")
        if not token_path:
            raise FileNotFoundError("token_path is required for AAEC evaluation")
        if not os.path.exists(token_path):
            raise FileNotFoundError(f"Gold file not found: {token_path}")
        return token_path
