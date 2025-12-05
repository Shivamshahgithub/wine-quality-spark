import sys
import os

from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler, StringIndexer
from pyspark.ml.classification import LogisticRegression, RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
import sys

def main(train_path, val_path, model_out):
    spark = SparkSession.builder.appName("WineTrain").getOrCreate()

    train_df = spark.read.option("header", True).option("inferSchema", True).csv(train_path)
    val_df   = spark.read.option("header", True).option("inferSchema", True).csv(val_path)

    label_col = "quality"
    feature_cols = [c for c in train_df.columns if c != label_col]

    assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")

    # If quality already numeric 1-10, you can skip indexer.
    indexer = StringIndexer(inputCol=label_col, outputCol="label", handleInvalid="skip")

    # Start simple: Logistic Regression (multinomial)
    lr = LogisticRegression(featuresCol="features", labelCol="label", maxIter=200, family="multinomial")

    pipeline = Pipeline(stages=[assembler, indexer, lr])
    model = pipeline.fit(train_df)

    preds = model.transform(val_df)

    evaluator = MulticlassClassificationEvaluator(
        labelCol="label", predictionCol="prediction", metricName="f1"
    )
    f1 = evaluator.evaluate(preds)
    print(f"Validation F1 = {f1}")

    model.write().overwrite().save(model_out)
    print(f"Saved model to {model_out}")

    spark.stop()

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: spark-submit train.py <train_csv> <val_csv> <model_out>")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2], sys.argv[3])

