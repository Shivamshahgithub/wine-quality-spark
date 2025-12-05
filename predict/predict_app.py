import sys
import os
from pyspark.sql import SparkSession
from pyspark.ml import PipelineModel
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
import sys

def main(model_path, test_path):
    spark = SparkSession.builder.appName("WinePredict").getOrCreate()

    df = spark.read.option("header", True).option("inferSchema", True).csv(test_path)

    model = PipelineModel.load(model_path)
    preds = model.transform(df)

    evaluator = MulticlassClassificationEvaluator(
        labelCol="label", predictionCol="prediction", metricName="f1"
    )
    f1 = evaluator.evaluate(preds)

    print(f"F1 score on test file = {f1}")

    spark.stop()

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: spark-submit predict.py <model_path> <test_csv>")
        sys.exit(1)

    main(sys.argv[1], sys.argv[2])
