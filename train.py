from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType, StringType, IntegerType
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler, StandardScaler

from src.data_preparation import stratified_split
from src.feature_engineering import feature_engineering_pipeline
from src.model_training import train_lgbm, optimize_threshold
from src.metrics import get_metrics
from src.utils import extract_features_from_vector

import numpy as np

DATA_PATH = ""

if __name__ == "__main__":
    # Load data
    spark = SparkSession.builder.appName("Incognia Chargeback Analysis").getOrCreate()
    raw_data = spark.read.parquet(DATA_PATH)

    data = raw_data.alias("data")
    data = data.withColumn("chargeback", F.col("chargeback").cast(StringType()))
    data = data.withColumn("label", F.when(F.col("chargeback") == "true", 1).otherwise(0))

    train_data, test_data = stratified_split(data, [0.8, 0.2], "label")
    train_data = feature_engineering_pipeline(train_data)
    test_data = feature_engineering_pipeline(test_data)
    numerical_cols = [
        "day_of_week",
        "day_of_month",
        "month",
        "seconds_from_midnight",
        "hour_of_day",
        "avg_chargeback_rate",
        "chargeback_rate_diff",
        "fraud_events_ratio",
        "cumulative_transaction_amount",
        "cumulative_fraud_count",
        "transaction_per_device_age",
        "fraud_per_user",
        "log_total_amount",
        "log_f_total_events",
        "log_f_total_fraud_events",
        "mean_transaction_amount",
        "max_transaction_amount",
        "stddev_transaction_amount"
    ]
    categorical_cols = ["day_period", "season"]

    # Prepare data for training
    X_train = train_data.select(*numerical_cols).toPandas()
    y_train = train_data.select("label").toPandas()

    X_test = test_data.select(*numerical_cols).toPandas()
    y_test = test_data.select("label").toPandas()

    # Optimize threshold and train model
    best_th, best_metrics = optimize_threshold(X_train, y_train, X_test, y_test, test_data_amounts.toPandas().values.ravel())
    print(f"Best Threshold: {best_th}, Metrics: {best_metrics}")
    
    # Train model with the best threshold
    metrics, predictions = train_lgbm(X_train, y_train, X_test, y_test, test_data_amounts.toPandas().values.ravel(), best_th)
    print(metrics)
