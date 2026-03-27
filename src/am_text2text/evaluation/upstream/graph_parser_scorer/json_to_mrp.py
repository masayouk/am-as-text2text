from typing import Dict, List, Optional, Tuple

from am_text2text.schemas.annotator import ParsedComponent as Component
from am_text2text.schemas.annotator import ParsedPrediction
from am_text2text.schemas.annotator import ParsedRelation as Relation

COMPONENT_LABEL_MAPPING = {
    "abstrct": {
        "major claim": "AbstRCT_MajorClaim",
        "claim": "AbstRCT_Claim",
        "premise": "AbstRCT_Evidence",
    },
    "abstruct": {
        "major claim": "AbstRCT_MajorClaim",
        "claim": "AbstRCT_Claim",
        "premise": "AbstRCT_Evidence",
    },
    "cdcp": {
        "value": "CDCP_value",
        "policy": "CDCP_policy",
        "testimony": "CDCP_testimony",
        "fact": "CDCP_fact",
        "reference": "CDCP_reference",
    },
}

RELATION_LABEL_MAPPING = {
    "abstrct": {
        "support": "AbstRCT_Support",
        "attack": "AbstRCT_Attack",
        "partial attack": "AbstRCT_Partial-Attack",
    },
    "abstruct": {
        "support": "AbstRCT_Support",
        "attack": "AbstRCT_Attack",
        "partial attack": "AbstRCT_Partial-Attack",
    },
    "cdcp": {
        "reason": "CDCP_reason",
        "evidence": "CDCP_evidence",
    },
}


def tokens_to_char_offsets(tokens: List[str]) -> List[Tuple[int, int]]:
    offsets = []
    current_pos = 0
    for token in tokens:
        start = current_pos
        end = current_pos + len(token)
        offsets.append((start, end))
        current_pos = end + 1
    return offsets


def tokens_to_char_offsets_in_text(tokens: List[str], input_text: str) -> List[Tuple[int, int]]:
    offsets = []
    search_pos = 0
    for token in tokens:
        idx = input_text.find(token, search_pos)
        if idx < 0:
            idx = input_text.lower().find(token.lower(), search_pos)
        if idx < 0:
            start = offsets[-1][1] + 1 if offsets else search_pos
            end = start + len(token)
        else:
            start = idx
            end = idx + len(token)
        offsets.append((start, end))
        search_pos = end
    return offsets


def json_to_mrp(
    tokens: List[str],
    components: List[Component],
    relations: List[Relation],
    doc_id: str,
    dataset: str,
    input_text: Optional[str] = None,
    token_offsets: Optional[List[Tuple[int, int]]] = None,
) -> Dict:
    comp_mapping = COMPONENT_LABEL_MAPPING.get(dataset, {})
    rel_mapping = RELATION_LABEL_MAPPING.get(dataset, {})

    if token_offsets is not None:
        pass
    elif input_text is None:
        input_text = " ".join(tokens)
        token_offsets = tokens_to_char_offsets(tokens)
    else:
        token_offsets = tokens_to_char_offsets(tokens) if input_text == " ".join(tokens) else tokens_to_char_offsets_in_text(tokens, input_text)

    nodes = []
    tops = []
    for i, comp in enumerate(components):
        if comp.start >= len(token_offsets) or comp.end > len(token_offsets) or comp.end <= 0:
            continue
        char_start = token_offsets[comp.start][0]
        char_end = token_offsets[comp.end - 1][1]
        label_lower = comp.type.lower()
        mrp_label = comp_mapping.get(label_lower, comp.type)
        nodes.append({"id": i, "label": mrp_label, "anchors": [{"from": char_start, "to": char_end}]})
        if "major" in label_lower or "majorclaim" in label_lower:
            tops.append(i)

    edges = []
    for rel in relations:
        if rel.source >= len(components) or rel.target >= len(components):
            continue
        rel_label_lower = rel.type.lower()
        mrp_rel_label = rel_mapping.get(rel_label_lower, rel.type)
        edges.append({"source": rel.source, "target": rel.target, "label": mrp_rel_label})

    if not tops and nodes:
        target_ids = {e["target"] for e in edges}
        for node in nodes:
            if node["id"] not in target_ids:
                tops.append(node["id"])

    return {
        "id": doc_id,
        "input": input_text,
        "framework": dataset,
        "flavor": 0,
        "nodes": nodes,
        "edges": edges,
        "tops": tops,
    }


def parsed_prediction_to_mrp(
    prediction: ParsedPrediction,
    tokens: List[str],
    doc_id: str,
    dataset: str,
    input_text: Optional[str] = None,
    token_offsets: Optional[List[Tuple[int, int]]] = None,
) -> Dict:
    return json_to_mrp(
        tokens=tokens,
        components=prediction.components,
        relations=prediction.relations,
        doc_id=doc_id,
        dataset=dataset,
        input_text=input_text,
        token_offsets=token_offsets,
    )
