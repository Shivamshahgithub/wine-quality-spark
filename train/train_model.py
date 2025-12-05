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

    # ---- Load data ----
    print("[INFO] Reading training CSV...")
    train_df = spark.read.csv(train_path, header=True, sep=";", inferSchema=True)
    
    print("[DEBUG] Training columns:", train_df.columns)
    train_df.printSchema()
    
    print("[INFO] Reading validation CSV...")
    val_df = spark.read.csv(val_path, header=True, sep=";", inferSchema=True)
    
    print("[DEBUG] Validation columns:", val_df.columns)
    val_df.printSchema()


    # label is the "quality" column
    label_col = "quality"
    feature_cols = [c for c in train_df.columns if c != label_col]

    # ---- Build pipeline ----
    indexer = StringIndexer(inputCol=label_col, outputCol="label", handleInvalid="keep")
    assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
    rf = RandomForestClassifier(
        labelCol="label",
        featuresCol="features",
        numTrees=50,
        maxDepth=10,
        seed=42,
    )

    pipeline = Pipeline(stages=[indexer, assembler, rf])

    # ---- Train ----
    print("[INFO] Fitting model (this can take ~30–60 seconds on first run)...")
    model = pipeline.fit(train_df)

    # ---- Evaluate ----
    print("[INFO] Evaluating on validation set...")
    preds = model.transform(val_df)

    evaluator = MulticlassClassificationEvaluator(
        labelCol="label",
        predictionCol="prediction",
        metricName="f1",
    )
    f1 = evaluator.evaluate(preds)
    print(f"[RESULT] Validation F1 score = {f1:.4f}")

    # ---- Save model ----
    print(f"[INFO] Saving model into directory: {model_dir}")
    os.makedirs(model_dir, exist_ok=True)
    model.write().overwrite().save(model_dir)

    abs_path = os.path.abspath(model_dir)
    print(f"[INFO] Model saved at: {abs_path}")

    spark.stop()
    print("[INFO] Training script finished successfully.")


if __name__ == "__main__":
    main(sys.argv)
