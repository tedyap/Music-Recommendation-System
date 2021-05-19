#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Usage:
    $ spark-submit train_tune.py
"""


from pyspark.sql import SparkSession
from pyspark.ml.feature import StringIndexer
from pyspark.ml.recommendation import ALS
from pyspark.sql import functions as F
from pyspark.mllib.evaluation import RankingMetrics


def get_data(spark, file_name, frac_keep):
    df = spark.read.parquet(f'hdfs:/user/bm106/pub/MSD/{file_name}.parquet')
    df = df.sample(False, frac_keep, 1)
    return df


def main(spark, SUBSET_SIZE):
    '''Main routine for Final Project
    Parameters
    ----------
    spark : SparkSession object
    netID : string, netID of student to find files in HDFS
    '''
    print('Loading in files')

    df_train = get_data(spark, 'cf_train_new', SUBSET_SIZE)
    print((df_train.count(), len(df_train.columns)))

    df_val = get_data(spark, 'cf_validation', SUBSET_SIZE)
    df_test = get_data(spark, 'cf_test', SUBSET_SIZE)

    # StringIndexing
    for column in ['user', 'track']:
        indexer = StringIndexer(inputCol=f'{column}_id', outputCol=f'{column}_idx', handleInvalid='keep')
        indexed = indexer.fit(df_train)
        df_train = indexed.transform(df_train)
        df_val = indexed.transform(df_val)
        df_test = indexed.transform(df_test)

    ranks = [5, 10, 15]
    regs = [0.1, 1, 10]

    best_map = 0
    best_rnk = 0
    best_reg = 0
    als_model_best = None

    # user's true recommendation
    true_recs = df_val.groupBy('user_idx').agg(F.collect_list('track_idx').alias('track_idx'))
    user_subset = df_val.select('user_idx').distinct()

    for rnk in ranks:
        for reg in regs:
            als = ALS(rank=rnk, regParam=reg, implicitPrefs=True, ratingCol="count", userCol="user_idx", itemCol="track_idx", coldStartStrategy="drop")
            als_model = als.fit(df_train)

            # user's predicted recommendation
            pred_recs = als_model.recommendForUserSubset(user_subset, 500)
            pred_recs = pred_recs.select('user_idx', F.col('recommendations.track_idx'))

            pred_true_rdd = pred_recs.join(true_recs, on='user_idx').rdd.map(lambda row: (row[1], row[2]))
            metrics = RankingMetrics(pred_true_rdd)
            map_ = metrics.meanAveragePrecision
            ndcg = metrics.ndcgAt(500)
            mpa = metrics.precisionAt(500)
            if map_ > best_map:
                best_rnk = rnk
                best_reg = reg
                als_model_best = als_model
            print('rank: {} reg: {} map: {} ndcg: {} mpa:{}'.format(rnk, reg, map_, ndcg, mpa))

        # user's true recommendation
        true_recs = df_test.groupBy('user_idx').agg(F.collect_list('track_idx').alias('track_idx'))

        # user's predicted recommendation
        user_subset = df_test.select('user_idx').distinct()
        pred_recs = als_model_best.recommendForUserSubset(user_subset, 500)
        pred_recs = pred_recs.select('user_idx', F.col('recommendations.track_idx'))

        pred_true_rdd = pred_recs.join(true_recs, on='user_idx').rdd.map(lambda row: (row[1], row[2]))
        metrics = RankingMetrics(pred_true_rdd)
        map_ = metrics.meanAveragePrecision
        ndcg = metrics.ndcgAt(500)
        mpa = metrics.precisionAt(500)
        print('Testing Dataset')
        print('rank: {} reg: {} map: {} ndcg: {} mpa:{}'.format(best_rnk, best_reg, map_, ndcg, mpa))


# Only enter this block if we're in main
if __name__ == "__main__":

    # Create the spark session object
    spark = SparkSession.builder.config('spark.sql.autoBroadcastJoinThreshold', '-1').appName('part1').getOrCreate()

    SUBSET_SIZE = 1.0
    # Call our main routine
    main(spark, SUBSET_SIZE)