from pyspark.sql import functions as F
from pyspark.sql.functions import when
from pyspark.sql.window import Window
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler, StandardScaler
from pyspark.ml import Pipeline

from src.data_preparation import impute_geohash_features_with_median


def create_temporal_features(data):
    data = data.withColumn('day_of_week', F.dayofweek('dt'))
    data = data.withColumn('day_of_month', F.dayofmonth('dt'))
    data = data.withColumn('month', F.month('dt'))
    data = data.withColumn('season', when((data.month >= 3) & (data.month <= 5), 'Spring')
                            .when((data.month >= 6) & (data.month <= 8), 'Summer')
                            .when((data.month >= 9) & (data.month <= 11), 'Autumn')
                            .otherwise('Winter'))
    data = data.withColumn('seconds_from_midnight', F.unix_timestamp('timestamp') % 86400)
    data = data.withColumn('hour_of_day', F.hour('timestamp'))
    data = data.withColumn(
        'day_period', 
        when((data['hour_of_day'] >= 0) & (data['hour_of_day'] < 6), 'Madrugada')
        .when((data['hour_of_day'] >= 6) & (data['hour_of_day'] < 12), 'Manhã')
        .when((data['hour_of_day'] >= 12) & (data['hour_of_day'] < 18), 'Tarde')
        .otherwise('Noite')
    )
    return data


def create_geographic_features(df):
    df = df.withColumn('mean_chargeback_rate', 
                       (F.col('f_chargeback_rate_by_geohash_5_30d') + 
                        F.col('f_chargeback_rate_by_geohash_6_30d') + 
                        F.col('f_chargeback_rate_by_geohash_7_30d')) / 3)
    
    df = df.withColumn('difference_high_low_geohash', 
                       F.abs(F.col('f_chargeback_rate_by_geohash_7_30d') - F.col('f_chargeback_rate_by_geohash_5_30d')))
    return df


def create_temporal_aggregations(df):
    window_spec = Window.partitionBy("device_id").orderBy(F.col("timestamp").cast("long")).rangeBetween(-30 * 86400, 0)
    df = df.withColumn("sum_values_last_30d", F.sum("total_amount").over(window_spec))
    # df = df.withColumn("stddev_values_last_30d", F.stddev("total_amount").over(window_spec))

    window_spec = Window.partitionBy("device_id").orderBy(F.col("timestamp").cast("long")).rangeBetween(-7 * 86400, 0)
    df = df.withColumn("transaction_count_last_30_days", F.count("id").over(window_spec))
    return df


def feature_preparation(df, numerical_cols, categorical_cols, label_col, total_amount_col):
    df = create_temporal_features(df)
    df = create_temporal_aggregations(df)
    df = impute_geohash_features_with_median(df)
    df = create_geographic_features(df)

    train_data, test_data = df.randomSplit([0.8, 0.2])

    train_data_amounts = train_data.select(total_amount_col)
    test_data_amounts = test_data.select(total_amount_col)
    
    train_data = train_data.drop(total_amount_col)
    test_data = test_data.drop(total_amount_col)

    stages = []
    for col in categorical_cols:
        indexer = StringIndexer(inputCol=col, outputCol=f"{col}_indexed")
        stages.append(indexer)
    
    assembler_numeric = VectorAssembler(inputCols=numerical_cols, outputCol="numerical_features")
    scaler = StandardScaler(inputCol="numerical_features", outputCol="scaled_numerical_features")
    stages += [assembler_numeric, scaler]

    feature_cols = [f"{col}_indexed" for col in categorical_cols] + ["scaled_numerical_features"]
    assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
    stages.append(assembler)

    pipeline = Pipeline(stages=stages)
    pipeline_model = pipeline.fit(train_data)

    train_data_transformed = pipeline_model.transform(train_data)
    test_data_transformed = pipeline_model.transform(test_data)
    return train_data_transformed, test_data_transformed, train_data_amounts, test_data_amounts