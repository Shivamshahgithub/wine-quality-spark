from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler, StringIndexer
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
import sys


def main(train_path, val_path, model_out):
    spark = (
        SparkSession.builder
        .appName("CS643-Wine-Train")
        .getOrCreate()
    )

    # Read CSVs (semicolon separator, header row)
    train_df = (
        spark.read
        .option("header", True)
        .option("inferSchema", True)
        .option("sep", ";")
        .csv(train_path)
    )
    val_df = (
        spark.read
        .option("header", True)
        .option("inferSchema", True)
        .option("sep", ";")
        .csv(val_path)
    )

    # Show raw column names (with crazy quotes)
    print("Raw train columns:", train_df.columns)
    print("Raw val columns  :", val_df.columns)

    # Rename columns in BOTH dataframes to simple, consistent names.
    # Assumption: column order is the same in train and val.
    train_cols = train_df.columns
    val_cols = val_df.columns
    if len(train_cols) != len(val_cols):
        raise ValueError(f"Train/val col length mismatch: {len(train_cols)} vs {len(val_cols)}")

    def rename_cols(df, cols):
        n = len(cols)
        renamed = df
        for idx, old in enumerate(cols):
            if idx == n - 1:
                new = "quality"          # last column = label
            else:
                new = f"f{idx}"          # f0, f1, ..., f10
            if old != new:
                renamed = renamed.withColumnRenamed(old, new)
        return renamed

    train_df = rename_cols(train_df, train_cols)
    val_df   = rename_cols(val_df, val_cols)

    print("Clean train columns:", train_df.columns)
    print("Clean val columns  :", val_df.columns)

    # After renaming we know exactly what the label/feature columns are
    label_col = "quality"
    feature_cols = [c for c in train_df.columns if c != label_col]

    # Index labels (1–10) to 0..(n-1)
    label_indexer = StringIndexer(
        inputCol=label_col,
        outputCol="label",
        handleInvalid="keep"
    )

    assembler = VectorAssembler(
        inputCols=feature_cols,
        outputCol="features"
    )

    evaluator = MulticlassClassificationEvaluator(
        labelCol="label",
        predictionCol="prediction",
        metricName="f1"
    )

    # Simple hyperparameter tuning over regParam + maxIter
    reg_params = [0.01, 0.1, 0.5]
    iters = [50, 100]

    best_f1 = -1.0
    best_model = None
    best_params = None

    for reg in reg_params:
        for it in iters:
            lr = LogisticRegression(
                featuresCol="features",
                labelCol="label",
                maxIter=it,
                regParam=reg,
                elasticNetParam=0.0
            )

            pipeline = Pipeline(stages=[label_indexer, assembler, lr])

            # Fit on training data
            model = pipeline.fit(train_df)

            # Evaluate on validation data
            preds = model.transform(val_df)
            f1 = evaluator.evaluate(preds)

            print(f"PARAMS regParam={reg}, maxIter={it}, F1={f1}")
            sys.stdout.flush()

            if f1 > best_f1:
                best_f1 = f1
                best_model = model
                best_params = (reg, it)

    print(f"BEST_PARAMS regParam={best_params[0]}, maxIter={best_params[1]}")
    print(f"BEST_F1={best_f1}")
    sys.stdout.flush()

    # Save best model (PipelineModel) to S3
    best_model.write().overwrite().save(model_out)

    spark.stop()


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: spark-submit train.py <train_path> <val_path> <model_out>")
        sys.exit(1)

    train_path = sys.argv[1]
    val_path = sys.argv[2]
    model_out = sys.argv[3]

    main(train_path, val_path, model_out)
