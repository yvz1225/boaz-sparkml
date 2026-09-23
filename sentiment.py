"""Hugging Face 문장 임베딩과 Spark ML로 영화 리뷰 감성을 분류한다."""

from __future__ import annotations

import argparse
import csv
import math
import os
import statistics
from pathlib import Path
from typing import Any

import mlflow
import mlflow.spark
from sentence_transformers import SentenceTransformer
from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import (
    BinaryClassificationEvaluator,
    MulticlassClassificationEvaluator,
)
from pyspark.ml.linalg import VectorUDT, Vectors
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import (
    DoubleType,
    StringType,
    StructField,
    StructType,
)


MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
MLFLOW_EXPERIMENT = "MovieReviewSentimentIMDb"
MAX_ITER = 50
REG_PARAMS = [0.0, 0.01, 0.1, 1.0]
VALIDATION_SEEDS = [42, 123, 2026]
NEW_REVIEWS = [
    "This movie was absolutely amazing and I loved it.",
    "The movie was boring and the story was terrible.",
    "The actors were great but the story was disappointing.",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="영화 리뷰 감성 분류 실습")
    parser.add_argument("--data", default="/app/data/MovieReviews.csv")
    parser.add_argument("--output", default="/app/output")
    parser.add_argument("--model", default=MODEL_NAME)
    return parser.parse_args()


def create_spark() -> SparkSession:
    spark = (
        SparkSession.builder.appName("MovieReviewSentiment")
        .master("local[*]")
        .config("spark.ui.enabled", "true")
        .config("spark.sql.shuffle.partitions", "2")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")
    return spark


def load_data(spark: SparkSession, path: str) -> DataFrame:
    raw = (
        spark.read.option("header", True)
        .option("inferSchema", True)
        .option("multiLine", True)
        .option("escape", '"')
        .csv(path)
    )
    if not {"text", "label", "split"}.issubset(raw.columns):
        raise ValueError(f"필수 열 text, label, split이 없습니다: {raw.columns}")

    df = raw.select(
        F.trim("text").alias("text"),
        F.col("label").cast("double").alias("label"),
        F.lower(F.trim("split")).alias("split"),
    )
    invalid = df.filter(
        F.col("text").isNull()
        | (F.length("text") == 0)
        | F.col("label").isNull()
        | (~F.col("label").isin(0.0, 1.0))
        | (~F.col("split").isin("train", "test"))
    ).count()
    if invalid:
        raise ValueError(f"빈 리뷰 또는 잘못된 label을 포함한 행이 {invalid}개 있습니다.")
    return df.cache()


def require_two_classes(name: str, df: DataFrame) -> None:
    labels = {row.label for row in df.select("label").distinct().collect()}
    if labels != {0.0, 1.0}:
        raise ValueError(f"{name} 데이터에 두 클래스가 모두 필요합니다: {labels}")


def embed_labeled(
    spark: SparkSession, df: DataFrame, encoder: SentenceTransformer
) -> DataFrame:
    rows = df.orderBy("text").collect()
    texts = [row.text for row in rows]
    vectors = encoder.encode(
        texts,
        batch_size=32,
        show_progress_bar=False,
        normalize_embeddings=True,
    )
    schema = StructType(
        [
            StructField("text", StringType(), False),
            StructField("label", DoubleType(), False),
            StructField("features", VectorUDT(), False),
        ]
    )
    embedded = spark.createDataFrame(
        [
            (row.text, float(row.label), Vectors.dense(vector))
            for row, vector in zip(rows, vectors)
        ],
        schema,
    )
    return embedded.cache()


def embed_texts(
    spark: SparkSession, texts: list[str], encoder: SentenceTransformer
) -> DataFrame:
    vectors = encoder.encode(
        texts,
        batch_size=32,
        show_progress_bar=False,
        normalize_embeddings=True,
    )
    schema = StructType(
        [
            StructField("text", StringType(), False),
            StructField("features", VectorUDT(), False),
        ]
    )
    embedded = spark.createDataFrame(
        [
            (text, Vectors.dense(vector))
            for text, vector in zip(texts, vectors)
        ],
        schema,
    )
    return embedded


def evaluate(predictions: DataFrame) -> dict[str, float]:
    multiclass = lambda metric: MulticlassClassificationEvaluator(
        labelCol="label", predictionCol="prediction", metricName=metric
    ).evaluate(predictions)
    auc = BinaryClassificationEvaluator(
        labelCol="label",
        rawPredictionCol="rawPrediction",
        metricName="areaUnderROC",
    ).evaluate(predictions)
    losses = []
    for row in predictions.select("label", "probability").collect():
        positive_probability = min(max(float(row.probability[1]), 1e-15), 1 - 1e-15)
        losses.append(
            -math.log(positive_probability)
            if row.label == 1.0
            else -math.log(1 - positive_probability)
        )
    return {
        "accuracy": multiclass("accuracy"),
        "f1": multiclass("f1"),
        "auc": auc,
        "log_loss": statistics.fmean(losses),
    }


def validation_summary(train_df: DataFrame, reg_param: float) -> dict[str, float]:
    seed_scores = []
    for seed in VALIDATION_SEEDS:
        fold_train, fold_validation = train_df.randomSplit([0.8, 0.2], seed=seed)
        require_two_classes(f"seed={seed} 학습", fold_train)
        require_two_classes(f"seed={seed} 검증", fold_validation)
        classifier = LogisticRegression(
            featuresCol="features",
            labelCol="label",
            maxIter=MAX_ITER,
            regParam=reg_param,
        )
        model = Pipeline(stages=[classifier]).fit(fold_train)
        seed_scores.append(evaluate(model.transform(fold_validation)))

    summary = {}
    for metric in ("accuracy", "f1", "auc", "log_loss"):
        values = [scores[metric] for scores in seed_scores]
        summary[f"validation_{metric}_mean"] = statistics.fmean(values)
        summary[f"validation_{metric}_std"] = statistics.pstdev(values)
    return summary


def train_models(
    train_df: DataFrame, test_df: DataFrame
) -> tuple[list[dict[str, float]], dict[float, Any], dict[float, DataFrame]]:
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(MLFLOW_EXPERIMENT)
    print(f"MLflow tracking: {tracking_uri} / experiment={MLFLOW_EXPERIMENT}")

    metrics, models, predictions = [], {}, {}
    train_count, test_count = train_df.count(), test_df.count()
    for reg_param in REG_PARAMS:
        with mlflow.start_run(run_name=f"regParam_{reg_param}"):
            validation = validation_summary(train_df, reg_param)
            classifier = LogisticRegression(
                featuresCol="features",
                labelCol="label",
                maxIter=MAX_ITER,
                regParam=reg_param,
            )
            model = Pipeline(stages=[classifier]).fit(train_df)
            result = model.transform(test_df).cache()
            scores = evaluate(result)

            mlflow.log_params(
                {
                    "dataset": "Stanford aclImdb",
                    "regParam": reg_param,
                    "maxIter": MAX_ITER,
                    "validationSeeds": ",".join(map(str, VALIDATION_SEEDS)),
                    "trainRows": train_count,
                    "testRows": test_count,
                }
            )
            mlflow.log_metrics({**scores, **validation})
            mlflow.spark.log_model(model, artifact_path="model")

        metrics.append({"regParam": reg_param, **scores, **validation})
        models[reg_param] = model
        predictions[reg_param] = result
    return metrics, models, predictions


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_results(
    output: str,
    metrics: list[dict[str, float]],
    test_predictions: DataFrame,
    new_predictions: DataFrame,
) -> None:
    output_dir = Path(output)
    write_csv(
        output_dir / "regparam_metrics.csv",
        [
            "regParam",
            "accuracy",
            "f1",
            "auc",
            "log_loss",
            "validation_accuracy_mean",
            "validation_accuracy_std",
            "validation_f1_mean",
            "validation_f1_std",
            "validation_auc_mean",
            "validation_auc_std",
            "validation_log_loss_mean",
            "validation_log_loss_std",
        ],
        [
            {
                key: row[key] if key == "regParam" else f"{row[key]:.6f}"
                for key in (
                    "regParam",
                    "accuracy",
                    "f1",
                    "auc",
                    "log_loss",
                    "validation_accuracy_mean",
                    "validation_accuracy_std",
                    "validation_f1_mean",
                    "validation_f1_std",
                    "validation_auc_mean",
                    "validation_auc_std",
                    "validation_log_loss_mean",
                    "validation_log_loss_std",
                )
            }
            for row in metrics
        ],
    )

    test_rows = test_predictions.select(
        "text", "label", "prediction", "probability"
    ).collect()
    write_csv(
        output_dir / "test_predictions_regparam_0_01.csv",
        ["text", "label", "prediction", "positiveProbability"],
        [
            {
                "text": row.text,
                "label": int(row.label),
                "prediction": int(row.prediction),
                "positiveProbability": f"{float(row.probability[1]):.6f}",
            }
            for row in test_rows
        ],
    )

    new_rows = new_predictions.select("text", "prediction", "probability").collect()
    write_csv(
        output_dir / "new_review_predictions.csv",
        ["text", "prediction", "sentiment", "positiveProbability"],
        [
            {
                "text": row.text,
                "prediction": int(row.prediction),
                "sentiment": "Positive" if row.prediction == 1.0 else "Negative",
                "positiveProbability": f"{float(row.probability[1]):.6f}",
            }
            for row in new_rows
        ],
    )


def main() -> None:
    args = parse_args()
    spark = create_spark()
    try:
        print("=== Part 1. 영화 리뷰 데이터 확인 ===")
        df = load_data(spark, args.data)
        df.show(5, truncate=80)
        df.printSchema()

        train_df = df.filter(F.col("split") == "train").drop("split").cache()
        test_df = df.filter(F.col("split") == "test").drop("split").cache()
        require_two_classes("학습", train_df)
        require_two_classes("테스트", test_df)
        print(f"전체 데이터: {df.count()}개")
        print(f"학습 데이터: {train_df.count()}개")
        print(f"테스트 데이터: {test_df.count()}개")

        print("\n=== Part 2. Hugging Face embedding 생성 ===")
        encoder = SentenceTransformer(args.model, device="cpu")
        dimension = encoder.get_embedding_dimension()
        if dimension != 384:
            raise ValueError(f"예상 embedding 차원은 384이지만 실제 값은 {dimension}입니다.")
        train_features = embed_labeled(spark, train_df, encoder)
        test_features = embed_labeled(spark, test_df, encoder)
        print(f"Pretrained model: {args.model}")
        print(f"Embedding dimension: {dimension}")
        train_features.select("text", "label", "features").show(3, truncate=80)

        metrics, models, predictions = train_models(train_features, test_features)
        base_metrics = next(row for row in metrics if row["regParam"] == 0.01)

        print("\n=== Part 3. Logistic Regression 평가 (regParam=0.01) ===")
        predictions[0.01].select(
            "text", "label", "prediction", "probability"
        ).show(10, truncate=70)
        print(f'Accuracy: {base_metrics["accuracy"]:.4f}')
        print(f'F1-score: {base_metrics["f1"]:.4f}')
        print(f'AUC: {base_metrics["auc"]:.4f}')
        print(f'Log Loss: {base_metrics["log_loss"]:.4f}')

        print("\n=== Part 4. 새로운 리뷰 감성 예측 ===")
        new_features = embed_texts(spark, NEW_REVIEWS, encoder)
        new_predictions = models[0.01].transform(new_features)
        new_predictions.select("text", "prediction", "probability").show(
            truncate=False
        )

        print("\n=== Part 5. MLflow regParam 실험 비교 ===")
        print("regParam | Test Acc | Test F1 | Test AUC | Log Loss | Validation F1 mean±std")
        print("---------+----------+---------+----------+----------+-----------------------")
        for row in metrics:
            print(
                f'{row["regParam"]:>8g} | {row["accuracy"]:>8.4f} | '
                f'{row["f1"]:>7.4f} | {row["auc"]:>8.4f} | '
                f'{row["log_loss"]:>8.4f} | '
                f'{row["validation_f1_mean"]:.4f}±{row["validation_f1_std"]:.4f}'
            )
        best_f1 = max(row["validation_f1_mean"] for row in metrics)
        best_params = [
            row["regParam"]
            for row in metrics
            if row["validation_f1_mean"] == best_f1
        ]
        if len(best_params) == 1:
            print(f"\n가장 높은 검증 평균 F1-score: regParam={best_params[0]:g}")
        else:
            print(f"\n가장 높은 검증 평균 F1-score는 {best_params}에서 동일합니다.")
        print("regularization을 크게 만든다고 테스트 성능이 항상 좋아지지는 않습니다.")
        print("MLflow UI: http://localhost:5000")

        save_results(args.output, metrics, predictions[0.01], new_predictions)
        print(f"결과 저장 완료: {args.output}")
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
