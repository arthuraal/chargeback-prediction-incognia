from pyspark.sql import functions as F
from pyspark.sql.window import Window

SEED = 42

def stratified_split(df, ratios: list=[0.8, 0.1, 0.1], target_col: str='label'):
    pos = df.filter(F.col(target_col) == 1)
    neg = df.filter(F.col(target_col) == 0)
    
    train_pos, valid_pos, test_pos = pos.randomSplit(ratios, seed=SEED)
    train_neg, valid_neg, test_neg = neg.randomSplit(ratios, seed=SEED)
    
    return train_pos.union(train_neg), valid_pos.union(valid_neg), test_pos.union(test_neg)

def impute_geohash_features_with_median(df):
    for col_name in ['f_chargeback_rate_by_geohash_5_30d', 'f_chargeback_rate_by_geohash_6_30d', 'f_chargeback_rate_by_geohash_7_30d']:
        median_value = df.stat.approxQuantile(col_name, [0.5], 0.01)[0]
        df = df.withColumn(col_name, F.when(F.col(col_name).isNull(), median_value).otherwise(F.col(col_name)))
    return df