import json
import re
from pathlib import Path

from am_text2text.evaluation.aaec_evaluator import AAECEvaluator
from am_text2text.evaluation.mrp_evaluator import MRPEvaluator
from am_text2text.evaluation.parser import PredictionParser
from am_text2text.schemas.annotator import EvaluationConfig, EvaluationMetrics, PredictionBatch
from am_text2text.schemas.dataset import CanonicalDataset, DatasetSourceConfig, DocumentRecord


def parse_prediction_batch(*, dataset: CanonicalDataset, split: str, raw_prediction_path: Path, output_format: str) -> PredictionBatch:
    payload = json.loads(raw_prediction_path.read_text(encoding="utf-8"))
    raw_predictions = payload.get("predictions", payload) if isinstance(payload, dict) else payload
    if not isinstance(raw_predictions, list):
        raise ValueError("prediction payload must be a list")
    documents = dataset.splits[split].documents
    if len(raw_predictions) != len(documents):
        raise ValueError(
            f"Prediction count mismatch for split={split}: predictions={len(raw_predictions)} docs={len(documents)}"
        )
    parser = PredictionParser()
    parsed_predictions = [parser.parse(str(raw_text), document, output_format=output_format) for raw_text, document in zip(raw_predictions, documents)]
    return PredictionBatch(
        dataset_name=dataset.dataset_name,
        split=split,
        raw_predictions=[str(text) for text in raw_predictions],
        parsed_predictions=parsed_predictions,
    )


def evaluate_prediction_batch(
    *,
    batch: PredictionBatch,
    source: DatasetSourceConfig,
    split: str,
    result_path: Path,
    config: EvaluationConfig,
    canonical_documents: list[DocumentRecord] | None = None,
) -> EvaluationMetrics:
    prediction_path = result_path.with_suffix(".predictions.json")
    prediction_path.parent.mkdir(parents=True, exist_ok=True)
    prediction_path.write_text(
        json.dumps([prediction.model_dump(mode="json") for prediction in batch.parsed_predictions], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    if source.source_type == "conll":
        evaluator = AAECEvaluator()
        result = evaluator.evaluate(
            prediction_path=str(prediction_path),
            result_output_path=str(result_path),
            split=split,
            dataset_name=source.dataset_name,
            token_path=str(source.token_path(split)),
            labels_path=str(source.label_path(split)),
            doc_ids=[prediction.doc_id for prediction in batch.parsed_predictions],
        )
    else:
        evaluator = MRPEvaluator(dataset=source.dataset_name)
        result = evaluator.evaluate(
            prediction_path=str(prediction_path),
            result_output_path=str(result_path),
            split=split,
            dataset_name=source.dataset_name,
            mrp_path=str(source.mrp_path(split)),
            overlap=config.overlap,
            space=config.space,
            canonical_documents=canonical_documents,
        )
    metrics_payload = result["metrics"]
    return EvaluationMetrics.model_validate(metrics_payload)


def list_checkpoints(model_dir: Path) -> list[int]:
    pattern = re.compile(r"checkpoint-(\d+)$")
    checkpoints: list[int] = []
    for child in model_dir.iterdir():
        match = pattern.search(child.name)
        if child.is_dir() and match:
            checkpoints.append(int(match.group(1)))
    return sorted(checkpoints)
