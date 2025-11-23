import sys
from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler, StringIndexer
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator


def main():
    if len(sys.argv) != 4:
        print("Usage: spark-submit train_model.py <training_csv> <validation_csv> <model_output_path>")
        sys.exit(1)

    train_path = sys.argv[1]
    val_path = sys.argv[2]
    model_output_path = sys.argv[3]

    spark = (
        SparkSession.builder
        .appName("WineQualityTraining")
        .getOrCreate()
    )

    # ---------- Load data ----------
    train_df = (
        spark.read.csv(
            train_path,
            header=True,
            sep=";",
            inferSchema=True
        )
    )

    val_df = (
        spark.read.csv(
            val_path,
            header=True,
            sep=";",
            inferSchema=True
        )
    )

    # Label column is "quality"
    label_col = "quality"

    # All other columns are features
    feature_cols = [c for c in train_df.columns if c != label_col]

    # ---------- ML pipeline ----------
    label_indexer = StringIndexer(
        inputCol=label_col,
        outputCol="label",
        handleInvalid="keep"
    )

    assembler = VectorAssembler(
        inputCols=feature_cols,
        outputCol="features"
    )

    rf = RandomForestClassifier(
        labelCol="label",
        featuresCol="features",
        numTrees=100,
        maxDepth=10,
        seed=42
    )

    pipeline = Pipeline(stages=[label_indexer, assembler, rf])

    # ---------- Train ----------
    model = pipeline.fit(train_df)

    # ---------- Evaluate on validation set ----------
    preds = model.transform(val_df)

    evaluator = MulticlassClassificationEvaluator(
        labelCol="label",
        predictionCol="prediction",
        metricName="f1"
    )

    f1 = evaluator.evaluate(preds)
    print(f"Validation F1 score: {f1}")

    # ---------- Save model ----------
    model.write().overwrite().save(model_output_path)

    spark.stop()


if __name__ == "__main__":
    main()
