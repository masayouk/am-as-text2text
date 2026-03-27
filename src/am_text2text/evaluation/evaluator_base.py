import json
import logging
import os
from abc import ABC, abstractmethod
from typing import Any, Optional

from am_text2text.schemas.annotator import EvaluationMetrics, ParsedPrediction


class BaseEvaluator(ABC):
    @abstractmethod
    def parse_predictions(
        self,
        prediction_path: str,
        gold_path: str,
        **kwargs,
    ) -> tuple[list[ParsedPrediction], Any]:
        raise NotImplementedError

    @abstractmethod
    def evaluate_metrics(
        self,
        parsed_predictions: list[ParsedPrediction],
        gold_data: Any,
        **kwargs,
    ) -> EvaluationMetrics:
        raise NotImplementedError

    def format_results(
        self,
        metrics: EvaluationMetrics,
        prediction_name: str,
        split: str,
        checkpoint: Optional[int] = None,
    ) -> dict[str, Any]:
        comp_f1 = metrics.component.f1
        rel_f1 = metrics.relation.f1
        return {
            "prediction_name": prediction_name,
            "split": split,
            "checkpoint": checkpoint,
            "metrics": {
                "span": metrics.span.model_dump(),
                "component": metrics.component.model_dump(),
                "relation": metrics.relation.model_dump(),
                "average_f1": (rel_f1 + comp_f1) / 2,
            },
        }

    def evaluate(
        self,
        prediction_path: str,
        result_output_path: str,
        *,
        split: str,
        dataset_name: str,
        checkpoint: Optional[int] = None,
        base_data_dir: Optional[str] = None,
        **kwargs,
    ) -> dict[str, Any]:
        logging.info("=== Evaluation phase (split=%s, dataset=%s) ===", split, dataset_name)
        gold_path = self._get_gold_path(split, dataset_name, base_data_dir, **kwargs)
        parsed_predictions, gold_data = self.parse_predictions(
            prediction_path,
            gold_path,
            base_data_dir=base_data_dir,
            **kwargs,
        )
        metrics = self.evaluate_metrics(
            parsed_predictions,
            gold_data,
            split=split,
            dataset=dataset_name,
            **kwargs,
        )
        prediction_name = os.path.splitext(os.path.basename(result_output_path))[0]
        result = self.format_results(metrics, prediction_name, split, checkpoint)
        with open(result_output_path, "w", encoding="utf-8") as handle:
            json.dump(result, handle, ensure_ascii=False, indent=2)
        return result

    @abstractmethod
    def _get_gold_path(
        self,
        split: str,
        dataset_name: str,
        base_data_dir: Optional[str] = None,
        **kwargs,
    ) -> str:
        raise NotImplementedError
