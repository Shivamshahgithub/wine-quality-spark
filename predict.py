from pyspark.sql import SparkSession
from pyspark.ml import PipelineModel
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
import sys


def rename_cols(df):
    """
    Rename columns in the SAME WAY as train.py:
    - All feature columns become f0, f1, ..., f10
    - Last column becomes 'quality'
    This must match what the pipeline expects.
    """
    cols = df.columns
    n = len(cols)
    renamed = df
    for idx, old in enumerate(cols):
        if idx == n - 1:
            new = "quality"          # last column = label
        else:
            new = f"f{idx}"          # f0, f1, f2, ...
        if old != new:
            renamed = renamed.withColumnRenamed(old, new)
    return renamed


def main(model_path, test_path):
    spark = (
        SparkSession.builder
        .appName("CS643-Wine-Predict")
        .getOrCreate()
    )

    # Read test CSV (same format as training: ; separated, with header)
    test_df = (
        spark.read
        .option("header", True)
        .option("inferSchema", True)
        .option("sep", ";")
        .csv(test_path)
    )

    print("Raw test columns:", test_df.columns)

    # IMPORTANT: make column names match what the saved PipelineModel expects
    test_df = rename_cols(test_df)
    print("Clean test columns:", test_df.columns)

    # Load the trained PipelineModel (StringIndexer + VectorAssembler + LR)
    model = PipelineModel.load(model_path)

    # Run prediction
    preds = model.transform(test_df)

    # Evaluate F1 using indexed labels ('label') and 'prediction'
    evaluator = MulticlassClassificationEvaluator(
        labelCol="label",
        predictionCol="prediction",
        metricName="f1"
    )

    f1 = evaluator.evaluate(preds)

    # This is the required output of the application
    print(f"F1={f1}")

    spark.stop()


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: spark-submit predict.py <model_path> <test_csv_path>")
        sys.exit(1)

    model_path = sys.argv[1]
    test_path = sys.argv[2]
    main(model_path, test_path)
