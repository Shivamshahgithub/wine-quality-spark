from pyspark.sql import SparkSession
from pyspark.ml import PipelineModel
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
import sys

def main():
    if len(sys.argv) != 3:
        print("Usage: predict_app.py <test_csv> <model_path>")
        sys.exit(1)

    test_csv = sys.argv[1]
    model_path = sys.argv[2]

    spark = SparkSession.builder.appName("WineQualityPrediction").getOrCreate()

    # Load CSV
    test_df = spark.read.csv(test_csv, header=True, sep=';', inferSchema=True)

    # Load model
    model = PipelineModel.load(model_path)

    # Predict
    preds = model.transform(test_df)

    evaluator = MulticlassClassificationEvaluator(
        labelCol="label",
        predictionCol="prediction",
        metricName="f1"
    )

    f1 = evaluator.evaluate(preds)
    print(f"F1 Score on Test Data: {f1}")

    spark.stop()


if __name__ == "__main__":
    main()

