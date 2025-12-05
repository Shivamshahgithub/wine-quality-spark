import sys
import os

from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler, StringIndexer
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator


def get_spark():
    spark = (
        SparkSession.builder
        .appName("WineQualityTraining")
        .getOrCreate()
    )
    return spark


def main(argv):
    if len(argv) != 4:
        print("Usage: python train/train_model.py <train_csv> <val_csv> <model_dir>")
        sys.exit(1)

    train_path = argv[1]
    val_path = argv[2]
    model_dir = argv[3]

    print(f"[INFO] Training data:   {train_path}")
    print(f"[INFO] Validation data: {val_path}")
    print(f"[INFO] Model output:    {model_dir}")

    spark = get_spark()

    # ------------ LOAD DATA (normal comma‑CSV) ------------
    print("[INFO] Reading training CSV...")
    train_df = spark.read.csv(train_path, header=True, inferSchema=True)

    print("[INFO] Reading validation CSV...")
    val_df = spark.read.csv(val_path, header=True, inferSchema=True)

    print("[DEBUG] Training columns:", train_df.columns)
    train_df.printSchema()

    # ------------ LABEL + FEATURES ------------
    # Use the LAST column as the label (should be quality)
    label_col = train_df.columns[-1]
    feature_cols = train_df.columns[:-1]

    print("[INFO] Using label column:", repr(label_col))
    print("[INFO] Number of feature columns:", len(feature_cols))

    indexer = StringIndexer(
        inputCol=label_col,
        outputCol="label",
        handleInvalid="keep",
    )

    assembler = VectorAssembler(
        inputCols=feature_cols,
        outputCol="features",
    )

    rf = RandomForestClassifier(
        labelCol="label",
        featuresCol="features",
        numTrees=50,
        maxDepth=10,
        seed=42,
    )

    pipeline = Pipeline(stages=[indexer, assembler, rf])

    # ------------ TRAIN ------------
    print("[INFO] Fitting model (this can take ~30–60 seconds on first run)...")
    model = pipeline.fit(train_df)

    # ------------ EVALUATE ------------
    print("[INFO] Evaluating on validation set...")
    preds = model.transform(val_df)

    evaluator = MulticlassClassificationEvaluator(
        labelCol="label",
        predictionCol="prediction",
        metricName="f1",
    )
    f1 = evaluator.evaluate(preds)
    print(f"[RESULT] Validation F1 score = {f1:.4f}")

    # ------------ SAVE MODEL ------------
    print(f"[INFO] Saving model into directory: {model_dir}")
    os.makedirs(model_dir, exist_ok=True)
    model.write().overwrite().save(model_dir)
    abs_path = os.path.abspath(model_dir)
    print(f"[INFO] Model saved at: {abs_path}")

    spark.stop()
    print("[INFO] Training script finished successfully.")


if __name__ == "__main__":
    main(sys.argv)
