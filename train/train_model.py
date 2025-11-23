from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler, StringIndexer
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
import sys

def main():
    if len(sys.argv) != 4:
        print("Usage: train_model.py <training_csv> <validation_csv> <model_output_path>")
        sys.exit(1)

    training_path = sys.argv[1]
    validation_path = sys.argv[2]
    model_output_path = sys.argv[3]

    spark = SparkSession.builder.appName("WineQualityTraining").getOrCreate()

    # Read CSVs
    train_df = spark.read.csv(training_path, header=True, sep=';', inferSchema=True)
    val_df = spark.read.csv(validation_path, header=True, sep=';', inferSchema=True)

    # Features
    feature_cols = [c for c in train_df.columns if c != "quality"]

    assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")

    label_indexer = StringIndexer(inputCol="quality", outputCol="label")

    # Model
    model = RandomForestClassifier(featuresCol="features", labelCol="label", numTrees=50)

    # Pipeline
    pipeline = Pipeline(stages=[label_indexer, assembler, model])

    # Train
    trained_model = pipeline.fit(train_df)

    # Evaluate on validation
    predictions = trained_model.transform(val_df)

    evaluator = MulticlassClassificationEvaluator(
        labelCol="label",
        predictionCol="prediction",
        metricName="f1"
    )

    f1 = evaluator.evaluate(predictions)
    print(f"Validation F1 Score: {f1}")

    # Save model
    trained_model.write().overwrite().save(model_output_path)

    spark.stop()


if __name__ == "__main__":
    main()

