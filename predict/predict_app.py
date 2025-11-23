import sys
import os
from pyspark.sql import SparkSession
from pyspark.ml import PipelineModel
from pyspark.ml.evaluation import MulticlassClassificationEvaluator


def main():
    if len(sys.argv) != 2:
        print("Usage: python predict_app.py <test_csv_path>")
        sys.exit(1)

    test_path = sys.argv[1]

    # By default look for trained model in ./model
    model_path = os.environ.get("MODEL_PATH", "./model")

    spark = (
        SparkSession.builder
        .appName("WineQualityPrediction")
        .getOrCreate()
    )

    # ---------- Load test data ----------
    test_df = (
        spark.read.csv(
            test_path,
            header=True,
            sep=";",
            inferSchema=True
        )
    )

    # ---------- Load trained pipeline model ----------
    model = PipelineModel.load(model_path)

    # ---------- Predict ----------
    preds = model.transform(test_df)

    evaluator = MulticlassClassificationEvaluator(
        labelCol="label",        # created by StringIndexer inside the pipeline
        predictionCol="prediction",
        metricName="f1"
    )

    f1 = evaluator.evaluate(preds)
    print(f"Test F1 score: {f1}")

    spark.stop()


if __name__ == "__main__":
    main()
