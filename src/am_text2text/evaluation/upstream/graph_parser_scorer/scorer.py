"""MRP scoring utilities.

Adapted from the scorer implementation released in hitachi-nlp/graph_parser for
Morio et al., "End-to-end Argument Mining with Cross-corpora Multi-task
Learning" (TACL 2022).

Paper: https://aclanthology.org/2022.tacl-1.37/
GitHub: https://github.com/hitachi-nlp/graph_parser
Upstream license: CC BY-NC-SA 4.0
"""

import copy
import json
from typing import Dict, List, Optional, Set, Tuple


class Scorer:
    def __init__(self):
        self.s = 0
        self.g = 0
        self.c = 0

    def add(self, system: Set[Tuple], gold: Set[Tuple]):
        self.s += len(system)
        self.g += len(gold)
        self.c += len(gold & system)

    @property
    def p(self):
        return self.c / self.s if self.s else 0.0

    @property
    def r(self):
        return self.c / self.g if self.g else 0.0

    @property
    def f(self):
        p = self.p
        r = self.r
        return (2.0 * p * r) / (p + r) if p + r > 0 else 0.0

    def dump(self):
        return {"g": self.g, "s": self.s, "c": self.c, "p": self.p, "r": self.r, "f": self.f}


def read_mrp_file(path: str):
    with open(path, "r", encoding="utf-8") as handle:
        return [_normalize_mrp(json.loads(line)) for line in handle.readlines() if line.strip()]


def _normalize_mrp(mrp: Dict) -> Dict:
    normalized = copy.deepcopy(mrp)
    normalized.setdefault("nodes", [])
    normalized.setdefault("edges", [])
    normalized.setdefault("tops", [])
    return normalized


def eval_anchor(s_mrps: List[Dict], g_mrps: List[Dict]) -> Dict:
    scorer = Scorer()
    for s_mrp, g_mrp in zip(s_mrps, g_mrps):
        scorer.add(
            system={(s_mrp["id"], node["anchors"][0]["from"], node["anchors"][0]["to"]) for node in s_mrp["nodes"]},
            gold={(g_mrp["id"], node["anchors"][0]["from"], node["anchors"][0]["to"]) for node in g_mrp["nodes"]},
        )
    return scorer.dump()


def eval_top(s_mrps: List[Dict], g_mrps: List[Dict]) -> Dict:
    scorer = Scorer()
    for s_mrp, g_mrp in zip(s_mrps, g_mrps):
        s_nid2anc = {node["id"]: (node["anchors"][0]["from"], node["anchors"][0]["to"]) for node in s_mrp["nodes"]}
        g_nid2anc = {node["id"]: (node["anchors"][0]["from"], node["anchors"][0]["to"]) for node in g_mrp["nodes"]}
        scorer.add(
            system={(s_mrp["id"],) + s_nid2anc[top] for top in s_mrp["tops"]},
            gold={(g_mrp["id"],) + g_nid2anc[top] for top in g_mrp["tops"]},
        )
    return scorer.dump()


def eval_label(s_mrps: List[Dict], g_mrps: List[Dict]) -> Dict:
    labels = set()
    for g_mrp in g_mrps:
        labels |= {node["label"] for node in g_mrp["nodes"] if "label" in node}
    label_scores = {}
    for label in labels:
        scorer = Scorer()
        for s_mrp, g_mrp in zip(s_mrps, g_mrps):
            scorer.add(
                system={
                    (s_mrp["id"], node["anchors"][0]["from"], node["anchors"][0]["to"])
                    for node in s_mrp["nodes"]
                    if "label" in node and node["label"] == label
                },
                gold={
                    (g_mrp["id"], node["anchors"][0]["from"], node["anchors"][0]["to"])
                    for node in g_mrp["nodes"]
                    if "label" in node and node["label"] == label
                },
            )
        label_scores[label] = scorer.dump()

    scorer = Scorer()
    for s_mrp, g_mrp in zip(s_mrps, g_mrps):
        scorer.add(
            system={
                (s_mrp["id"], node["anchors"][0]["from"], node["anchors"][0]["to"], node["label"])
                for node in s_mrp["nodes"]
                if "label" in node
            },
            gold={
                (g_mrp["id"], node["anchors"][0]["from"], node["anchors"][0]["to"], node["label"])
                for node in g_mrp["nodes"]
                if "label" in node
            },
        )
    label_scores["total"] = scorer.dump()
    return label_scores


def eval_edge(s_mrps: List[Dict], g_mrps: List[Dict]) -> Dict:
    s_n2anc, g_n2anc = {}, {}
    for s_mrp, g_mrp in zip(s_mrps, g_mrps):
        for node in s_mrp["nodes"]:
            anchors = node.get("anchors") or []
            if anchors:
                anc = anchors[0]
                s_n2anc[(s_mrp["id"], node["id"])] = (anc["from"], anc["to"])
        for node in g_mrp["nodes"]:
            anchors = node.get("anchors") or []
            if anchors:
                anc = anchors[0]
                g_n2anc[(g_mrp["id"], node["id"])] = (anc["from"], anc["to"])

    edge_labels = set()
    for g_mrp in g_mrps:
        edge_labels |= {edge["label"] for edge in g_mrp["edges"] if "label" in edge}

    def _edge_tuple(edge: Dict, mapping: Dict, mrp_id: str) -> Optional[tuple]:
        source = mapping.get((mrp_id, edge.get("source")))
        target = mapping.get((mrp_id, edge.get("target")))
        if source is None or target is None:
            return None
        return (mrp_id, *source, *target)

    label_scores = {}
    for label in edge_labels:
        scorer = Scorer()
        for s_mrp, g_mrp in zip(s_mrps, g_mrps):
            scorer.add(
                system={
                    edge_tuple
                    for edge in s_mrp["edges"]
                    if "label" in edge and edge["label"] == label
                    if (edge_tuple := _edge_tuple(edge, s_n2anc, s_mrp["id"])) is not None
                },
                gold={
                    edge_tuple
                    for edge in g_mrp["edges"]
                    if "label" in edge and edge["label"] == label
                    if (edge_tuple := _edge_tuple(edge, g_n2anc, g_mrp["id"])) is not None
                },
            )
        label_scores[label] = scorer.dump()

    link_scorer = Scorer()
    for s_mrp, g_mrp in zip(s_mrps, g_mrps):
        link_scorer.add(
            system={
                edge_tuple
                for edge in s_mrp["edges"]
                if (edge_tuple := _edge_tuple(edge, s_n2anc, s_mrp["id"])) is not None
            },
            gold={
                edge_tuple
                for edge in g_mrp["edges"]
                if (edge_tuple := _edge_tuple(edge, g_n2anc, g_mrp["id"])) is not None
            },
        )
    label_scores["link"] = link_scorer.dump()

    scorer = Scorer()
    for s_mrp, g_mrp in zip(s_mrps, g_mrps):
        scorer.add(
            system={
                (
                    s_mrp["id"],
                    *s_n2anc[(s_mrp["id"], edge["source"])],
                    *s_n2anc[(s_mrp["id"], edge["target"])],
                    (edge["label"] if "label" in edge else ""),
                )
                for edge in s_mrp["edges"]
                if (s_mrp["id"], edge["source"]) in s_n2anc and (s_mrp["id"], edge["target"]) in s_n2anc
            },
            gold={
                (
                    g_mrp["id"],
                    *g_n2anc[(g_mrp["id"], edge["source"])],
                    *g_n2anc[(g_mrp["id"], edge["target"])],
                    (edge["label"] if "label" in edge else ""),
                )
                for edge in g_mrp["edges"]
                if (g_mrp["id"], edge["source"]) in g_n2anc and (g_mrp["id"], edge["target"]) in g_n2anc
            },
        )
    label_scores["total"] = scorer.dump()
    return label_scores


def relieve_overlap(s_mrp: Dict, g_mrp: Dict, overlap: float):
    s_mrp = copy.deepcopy(s_mrp)
    g_anchors = [n["anchors"][0] for n in g_mrp["nodes"]]
    for s_node in s_mrp["nodes"]:
        s_anc = s_node["anchors"][0]
        max_overlap_len = 0
        for g_anc in g_anchors:
            max_len = max(g_anc["to"] - g_anc["from"], s_anc["to"] - s_anc["from"])
            overlap_len = len(set(range(g_anc["from"], g_anc["to"])) & set(range(s_anc["from"], s_anc["to"])))
            overlap_rate = float(overlap_len) / max_len
            if overlap_rate > overlap and overlap_rate > max_overlap_len:
                s_node["anchors"][0] = g_anc
                max_overlap_len = overlap_rate
    return s_mrp


def relieve_space(s_mrp: Dict, g_mrp: Dict):
    s_mrp = copy.deepcopy(s_mrp)
    g_mrp = copy.deepcopy(g_mrp)
    txt = s_mrp["input"]
    for mrp in [s_mrp, g_mrp]:
        for node in mrp["nodes"]:
            while True:
                anc = node["anchors"][0]
                if txt[anc["from"]] == " ":
                    node["anchors"] = [{"from": anc["from"] + 1, "to": anc["to"]}]
                else:
                    break
            while True:
                anc = node["anchors"][0]
                if txt[anc["to"] - 1] == " ":
                    node["anchors"] = [{"from": anc["from"], "to": anc["to"] - 1}]
                else:
                    break
    return s_mrp, g_mrp


def add_top_scores_into_edge_scores(res_top: Dict, res_edge: Dict):
    res_edge = copy.deepcopy(res_edge)
    for edge_metric in ["total", "link"]:
        g, s, c = (res_top[metric] + res_edge[edge_metric][metric] for metric in ["g", "s", "c"])
        p = c / s if s else 0.0
        r = c / g if g else 0.0
        f = (2.0 * p * r) / (p + r) if p + r > 0 else 0.0
        res_edge[f"{edge_metric}-top-as-edge"] = {"g": g, "s": s, "c": c, "p": p, "r": r, "f": f}
    return res_edge
