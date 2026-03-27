import itertools
from collections import defaultdict
from typing import Set

import numpy as np

from am_text2text.schemas.annotator import ParsedComponent, ParsedPrediction, ParsedRelation
from am_text2text.schemas.dataset import DocumentRecord, resolve_dataset_annotation_schema


class PredictionParser:
    def parse(self, raw_text: str, document: DocumentRecord, *, output_format: str) -> ParsedPrediction:
        output_format = output_format.lower()
        if output_format != "tanl":
            raise ValueError(f"Unsupported annotation output format: {output_format}")
        return self._parse_tanl(raw_text, document)

    def _parse_tanl(self, raw_text: str, document: DocumentRecord) -> ParsedPrediction:
        annotation_schema = resolve_dataset_annotation_schema(document.dataset_name)
        parser = _TanlParser(
            entity_types=set(annotation_schema.entity_types),
            relation_types=set(annotation_schema.relation_types),
            example_tokens=document.tokens,
            output_sentence=_strip_fenced(raw_text),
        )
        entity_tuples, relation_tuples = parser.parse()
        ordered_entities = sorted(entity_tuples, key=lambda item: (item[1], item[2], item[0]))
        components = [
            ParsedComponent(type=annotation_schema.normalize_component_label(entity_type), start=start, end=end)
            for entity_type, start, end in ordered_entities
        ]
        span_to_index = {(entity_type, start, end): index for index, (entity_type, start, end) in enumerate(ordered_entities)}
        relations = []
        for rel_type, source_tuple, target_tuple in relation_tuples:
            source_index = span_to_index.get(source_tuple)
            target_index = span_to_index.get(target_tuple)
            if source_index is None or target_index is None:
                continue
            relations.append(
                ParsedRelation(
                    type=annotation_schema.normalize_relation_label(rel_type),
                    source=source_index,
                    target=target_index,
                )
            )
        return ParsedPrediction(doc_id=document.doc_id, components=components, relations=relations)

def _strip_fenced(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```") and stripped.endswith("```"):
        stripped = "\n".join(stripped.splitlines()[1:-1]).strip()
    return stripped

def _normalize_annotation_label(label: str) -> str:
    return " ".join(label.replace("_", " ").replace("-", " ").split()).lower()


class _TanlParser:
    START_TOKEN = "["
    END_TOKEN = "]"
    SEPARATOR_TOKEN = "|"
    EQUAL_TOKEN = "="

    def __init__(self, entity_types: Set[str], relation_types: Set[str], example_tokens: list[str], output_sentence: str):
        self.entity_types = entity_types
        self.relation_types = relation_types
        self.example_tokens = example_tokens
        self.output_sentence = output_sentence

    def parse(self):
        raw_entities, _ = self._parse_and_align_sentence()
        predicted_entities_by_name = defaultdict(list)
        predicted_entities = set()
        raw_relations = []
        for entity_name, tags, start, end in raw_entities:
            if not tags or len(tags[0]) > 1:
                continue
            entity_type = _normalize_annotation_label(tags[0][0])
            if entity_type not in self.entity_types:
                continue
            entity_tuple = (entity_type, start, end)
            predicted_entities.add(entity_tuple)
            predicted_entities_by_name[entity_name].append(entity_tuple)
            for tag in tags[1:]:
                if len(tag) == 2:
                    relation_type = _normalize_annotation_label(tag[0])
                    related_entity = tag[1]
                    if relation_type in self.relation_types:
                        raw_relations.append((relation_type, entity_tuple, related_entity))
        predicted_relations = set()
        for rel_type, annotated_entity, pointed_entity_name in raw_relations:
            if pointed_entity_name not in predicted_entities_by_name:
                continue
            _, annotated_start, annotated_end = annotated_entity
            candidates = sorted(
                predicted_entities_by_name[pointed_entity_name],
                key=lambda item: min(abs(item[1] - annotated_end), abs(annotated_start - item[2])),
            )
            if candidates:
                pointed_entity = candidates[0]
                predicted_relations.add((rel_type, pointed_entity, annotated_entity))
        return predicted_entities, predicted_relations

    def _parse_and_align_sentence(self):
        padded_output = self.output_sentence
        for token in [self.START_TOKEN, self.END_TOKEN, self.SEPARATOR_TOKEN, self.EQUAL_TOKEN]:
            padded_output = padded_output.replace(token, f" {token} ")

        entity_stack = []
        output_tokens = []
        unmatched_entities = []
        for token in padded_output.split():
            if token == self.START_TOKEN:
                entity_stack.append([len(output_tokens), "name", [], []])
            elif token == self.END_TOKEN and entity_stack:
                start, _, name_tokens, other_tokens = entity_stack.pop()
                entity_name = " ".join(name_tokens).strip()
                tags = [
                    tuple(" ".join(group).split(f" {self.EQUAL_TOKEN} "))
                    for is_sep, group in itertools.groupby(other_tokens, lambda item: item == self.SEPARATOR_TOKEN)
                    if not is_sep
                ]
                unmatched_entities.append((entity_name, tags, start, len(output_tokens)))
            elif token == self.SEPARATOR_TOKEN and entity_stack:
                if entity_stack[-1][1] == "name":
                    entity_stack[-1][1] = "other"
                else:
                    entity_stack[-1][3].append(token)
            elif entity_stack:
                is_name_token = True
                for entry in reversed(entity_stack):
                    if entry[1] == "name":
                        entry[2].append(token)
                    else:
                        entry[3].append(token)
                        is_name_token = False
                        break
                if is_name_token:
                    output_tokens.append(token)
            else:
                output_tokens.append(token)

        matching = self._align_tokens(self.example_tokens, output_tokens)
        predicted_entities = []
        for entity_name, entity_tags, start, end in unmatched_entities:
            new_start = None
            new_end = None
            for token_index in range(start, end):
                if token_index in matching:
                    if new_start is None:
                        new_start = matching[token_index]
                    new_end = matching[token_index]
            if new_start is not None and new_end is not None:
                predicted_entities.append((entity_name, entity_tags, new_start, new_end + 1))
        return predicted_entities, "".join(output_tokens) != "".join(self.example_tokens)

    @staticmethod
    def _align_tokens(tokens1: list[str], tokens2: list[str]):
        cost = np.zeros((len(tokens1) + 1, len(tokens2) + 1))
        best = np.zeros_like(cost, dtype=int)
        for i in range(len(tokens1) + 1):
            for j in range(len(tokens2) + 1):
                if i == 0 and j == 0:
                    continue
                candidates = []
                if i > 0 and j > 0:
                    mismatch = 0 if tokens1[i - 1] == tokens2[j - 1] else 1
                    candidates.append((mismatch + cost[i - 1, j - 1], 1))
                if i > 0:
                    candidates.append((1 + cost[i - 1, j], 2))
                if j > 0:
                    candidates.append((1 + cost[i, j - 1], 3))
                chosen_cost, chosen_option = min(candidates)
                cost[i, j] = chosen_cost
                best[i, j] = chosen_option

        matching = {}
        i, j = len(tokens1) - 1, len(tokens2) - 1
        while i >= 0 and j >= 0:
            option = best[i + 1, j + 1]
            if option == 1:
                matching[j] = i
                i -= 1
                j -= 1
            elif option == 2:
                i -= 1
            else:
                j -= 1
        return matching
