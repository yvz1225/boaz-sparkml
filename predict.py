"""MLflow에 저장된 Spark 모델로 새 영화 리뷰 하나를 예측한다."""

from __future__ import annotations

import argparse
import os

import mlflow
import mlflow.spark
from mlflow.tracking import MlflowClient
from sentence_transformers import SentenceTransformer

from sentiment import MLFLOW_EXPERIMENT, MODEL_NAME, create_spark, embed_texts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="새 영화 리뷰 감성 예측")
    parser.add_argument("text", help="예측할 영어 영화 리뷰")
    parser.add_argument(
        "--reg-param",
        default="0.01",
        choices=["0.0", "0.01", "0.1", "1.0"],
        help="MLflow에서 불러올 모델의 regParam (기본값: 0.01)",
    )
    parser.add_argument("--model", default=MODEL_NAME)
    return parser.parse_args()


def latest_run_id(reg_param: str) -> str:
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient()
    experiment = client.get_experiment_by_name(MLFLOW_EXPERIMENT)
    if experiment is None:
        raise RuntimeError("저장된 실험이 없습니다. sentiment.py를 먼저 실행하세요.")

    runs = client.search_runs(
        [experiment.experiment_id],
        filter_string=(
            f"attributes.status = 'FINISHED' and params.regParam = '{reg_param}'"
        ),
        order_by=["attributes.start_time DESC"],
        max_results=1,
    )
    if not runs:
        raise RuntimeError(
            f"regParam={reg_param} 모델이 없습니다. sentiment.py를 먼저 실행하세요."
        )
    return runs[0].info.run_id


def main() -> None:
    args = parse_args()
    spark = create_spark()
    try:
        run_id = latest_run_id(args.reg_param)
        encoder = SentenceTransformer(args.model, device="cpu")
        features = embed_texts(spark, [args.text], encoder)
        model = mlflow.spark.load_model(f"runs:/{run_id}/model")
        prediction = model.transform(features)

        print("\n=== Part 4. 새로운 영화 리뷰 감성 예측 ===")
        print(f"MLflow model: regParam={args.reg_param}, run_id={run_id}")
        prediction.select("text", "prediction", "probability").show(
            truncate=False
        )
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
