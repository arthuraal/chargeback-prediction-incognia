from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType, StringType, IntegerType
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler, StandardScaler

from src.data_preparation import stratified_split, impute_geohash_features_with_median
from src.feature_engineering import feature_preparation
from src.model_training import train_lgbm, optimize_threshold
from src.metrics import get_metrics
from src.utils import extract_features_from_vector

import numpy as np

DATA_PATH = ""

if __name__ == "__main__":
    # Load data
    spark = SparkSession.builder.appName("Incognia Chargeback Analysis").getOrCreate()
    raw_data = spark.read.parquet(DATA_PATH)
    raw_data = raw_data.withColumn("chargeback", F.col("chargeback").cast(StringType()))
    
    
    # Preprocess the data
    data = raw_data.alias("data")
    indexer_chargeback = StringIndexer(inputCol="chargeback", outputCol="label")
    data = indexer_chargeback.fit(data).transform(data)

    numerical_cols = ['mean_chargeback_rate', 'hour_of_day', 'seconds_from_midnight', 'day_of_week', 'day_of_month', 'month', 'transaction_count_last_30_days', 'sum_values_last_30d']
    categorical_cols = ['day_period', 'season']

    # Split data
    train_data, test_data, _, test_data_amounts = feature_preparation(data, numerical_cols, categorical_cols, "label", "total_amount")
    
    # Prepare data for training
    X_train = train_data.select("features").toPandas()
    y_train = train_data.select("label").toPandas().values.ravel()
    
    X_test = test_data.select("features").toPandas()
    y_test = test_data.select("label").toPandas().values.ravel()
    
    # Optimize threshold and train model
    best_th, best_metrics = optimize_threshold(X_train, y_train, X_test, y_test, test_data_amounts.toPandas().values.ravel())
    print(f"Best Threshold: {best_th}, Metrics: {best_metrics}")
    
    # Train model with the best threshold
    metrics, predictions = train_lgbm(X_train, y_train, X_test, y_test, test_data_amounts.toPandas().values.ravel(), best_th)
    print(metrics)
