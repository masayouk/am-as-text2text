import json
import os
from typing import Optional

from am_text2text.evaluation.evaluator_base import BaseEvaluator
from am_text2text.evaluation.upstream.graph_parser_scorer.json_to_mrp import (
    COMPONENT_LABEL_MAPPING,
    RELATION_LABEL_MAPPING,
    parsed_prediction_to_mrp,
)
from am_text2text.evaluation.upstream.graph_parser_scorer.scorer import (
    add_top_scores_into_edge_scores,
    eval_anchor,
    eval_edge,
    eval_label,
    eval_top,
    read_mrp_file,
    relieve_overlap,
    relieve_space,
)
from am_text2text.evaluation.tokenizer import tokenize_with_doc
from am_text2text.schemas.annotator import EvaluationMetrics, MetricScore, ParsedPrediction
from am_text2text.schemas.dataset import DocumentRecord


def evaluate_mrp(
    s_mrps: list[dict],
    g_mrps: list[dict],
    overlap: Optional[float] = None,
    space: bool = False,
    top_as_edge: bool = False,
) -> dict:
    if overlap is not None:
        s_mrps = [relieve_overlap(s_mrp=s_mrp, g_mrp=g_mrp, overlap=overlap) for s_mrp, g_mrp in zip(s_mrps, g_mrps)]
    if space:
        adjusted_system = []
        adjusted_gold = []
        for s_mrp, g_mrp in zip(s_mrps, g_mrps):
            new_s_mrp, new_g_mrp = relieve_space(s_mrp=s_mrp, g_mrp=g_mrp)
            adjusted_system.append(new_s_mrp)
            adjusted_gold.append(new_g_mrp)
        s_mrps = adjusted_system
        g_mrps = adjusted_gold

    res_top = eval_top(s_mrps=s_mrps, g_mrps=g_mrps)
    res_anchor = eval_anchor(s_mrps=s_mrps, g_mrps=g_mrps)
    res_label = eval_label(s_mrps=s_mrps, g_mrps=g_mrps)
    res_edge = eval_edge(s_mrps=s_mrps, g_mrps=g_mrps)
    if top_as_edge:
        res_edge = add_top_scores_into_edge_scores(res_top=res_top, res_edge=res_edge)
    return {"tops": res_top, "anchors": res_anchor, "labels": res_label, "edges": res_edge}


def _score_from_entry(entry: dict) -> MetricScore:
    return MetricScore(
        precision=entry["p"],
        recall=entry["r"],
        f1=entry["f"],
    )


def _mrp_results_to_metrics(mrp_results: dict) -> EvaluationMetrics:
    component = _score_from_entry(mrp_results["labels"]["total"])
    relation = _score_from_entry(mrp_results["edges"]["total"])
    return EvaluationMetrics(
        span=_score_from_entry(mrp_results["anchors"]),
        component=component,
        relation=relation,
        average_f1=(component.f1 + relation.f1) / 2,
    )


def _prediction_to_mrp_from_document(prediction: ParsedPrediction, document: DocumentRecord, dataset: str) -> dict:
    token_offsets = [(offset.start, offset.end) for offset in document.token_offsets]
    return parsed_prediction_to_mrp(
        prediction,
        document.tokens,
        document.doc_id,
        dataset,
        input_text=document.raw_text,
        token_offsets=token_offsets,
    )


def _prediction_to_mrp_from_gold(
    prediction: ParsedPrediction,
    gold_mrp: dict,
    dataset: str,
    *,
    index: int,
) -> dict:
    comp_mapping = COMPONENT_LABEL_MAPPING.get(dataset, {})
    rel_mapping = RELATION_LABEL_MAPPING.get(dataset, {})
    doc_id = gold_mrp.get("id", f"doc_{index}")
    input_text = gold_mrp.get("input", "")
    doc, tokens, offsets = tokenize_with_doc(input_text)
    max_len = len(doc) if doc is not None else len(tokens)

    nodes = []
    tops = []
    for comp_index, component in enumerate(prediction.components):
        if component.start >= max_len or component.end > max_len or component.end <= 0:
            continue
        if doc is not None:
            span = doc[component.start : component.end]
            char_start = span.start_char
            char_end = span.end_char
        else:
            char_start = offsets[component.start][0]
            char_end = offsets[component.end - 1][1]
        label_lower = component.type.lower()
        nodes.append(
            {
                "id": comp_index,
                "label": comp_mapping.get(label_lower, component.type),
                "anchors": [{"from": char_start, "to": char_end}],
            }
        )
        if "major" in label_lower or "majorclaim" in label_lower:
            tops.append(comp_index)

    edges = []
    for relation in prediction.relations:
        if relation.source >= len(prediction.components) or relation.target >= len(prediction.components):
            continue
        edges.append(
            {
                "source": relation.source,
                "target": relation.target,
                "label": rel_mapping.get(relation.type.lower(), relation.type),
            }
        )

    if not tops and nodes:
        tops = _top_nodes_from_edges(nodes, edges)

    return {
        "id": doc_id,
        "input": input_text,
        "framework": dataset,
        "flavor": 0,
        "nodes": nodes,
        "edges": edges,
        "tops": tops,
    }


def _top_nodes_from_edges(nodes: list[dict], edges: list[dict]) -> list[int]:
    target_ids = {edge["target"] for edge in edges}
    return [node["id"] for node in nodes if node["id"] not in target_ids]


class MRPEvaluator(BaseEvaluator):
    def __init__(self, dataset: str = "cdcp"):
        self.dataset = dataset

    def parse_predictions(self, prediction_path: str, gold_path: str, **kwargs):
        del kwargs
        with open(prediction_path, "r", encoding="utf-8") as handle:
            pred_data = json.load(handle)
        preds_list = pred_data.get("predictions", pred_data) if isinstance(pred_data, dict) else pred_data
        if not isinstance(preds_list, list):
            preds_list = [preds_list]
        parsed_predictions = [ParsedPrediction.model_validate(pred) for pred in preds_list]
        return parsed_predictions, gold_path

    def evaluate_metrics(self, parsed_predictions, gold_data, overlap=None, space=False, **kwargs):
        dataset = kwargs.get("dataset", self.dataset)
        g_mrps = read_mrp_file(gold_data) if isinstance(gold_data, str) else gold_data
        s_mrps = self._predictions_to_mrp(
            parsed_predictions,
            g_mrps,
            dataset,
            documents=kwargs.get("canonical_documents"),
        )
        mrp_results = evaluate_mrp(s_mrps=s_mrps, g_mrps=g_mrps, overlap=overlap, space=space)
        return _mrp_results_to_metrics(mrp_results)

    def _predictions_to_mrp(
        self,
        predictions: list[ParsedPrediction],
        g_mrps: list[dict],
        dataset: str,
        *,
        documents: list[DocumentRecord] | None = None,
    ) -> list[dict]:
        if documents is not None:
            return [
                _prediction_to_mrp_from_document(prediction, document, dataset)
                for prediction, document in zip(predictions, documents)
            ]
        return [
            _prediction_to_mrp_from_gold(prediction, gold_mrp, dataset, index=index)
            for index, (prediction, gold_mrp) in enumerate(zip(predictions, g_mrps))
        ]

    def _get_gold_path(self, split: str, dataset_name: str, base_data_dir: Optional[str] = None, **kwargs) -> str:
        del split, dataset_name, base_data_dir
        gold_path = kwargs.get("mrp_path")
        if not gold_path:
            raise FileNotFoundError("mrp_path is required for MRP evaluation")
        if not os.path.exists(gold_path):
            raise FileNotFoundError(f"MRP gold file not found: {gold_path}")
        return gold_path
