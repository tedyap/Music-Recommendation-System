#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Usage:
    $ spark-submit train_tune.py <student_netID>
"""


# And pyspark.sql to get the spark session
from pyspark.sql import SparkSession
from pyspark.ml.feature import StringIndexer
from pyspark.ml.recommendation import ALS
from pyspark.sql import functions as F
from pyspark.mllib.evaluation import RankingMetrics
import time


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

    df_test = get_data(spark, 'cf_test', SUBSET_SIZE)

    # StringIndexing
    for column in ['user', 'track']:
        indexer = StringIndexer(inputCol=f'{column}_id', outputCol=f'{column}_idx', handleInvalid='keep')
        indexed = indexer.fit(df_train)
        df_train = indexed.transform(df_train)
        df_test = indexed.transform(df_test)


    # user's true recommendation
    true_recs = df_test.groupBy('user_idx').agg(F.collect_list('track_idx').alias('track_idx'))

    # user's predicted recommendation
    user_subset = df_test.select('user_idx').distinct()

    start = time.time()
    als = ALS(rank=15, regParam=0.1, implicitPrefs=True, ratingCol="count", userCol="user_idx", itemCol="track_idx", coldStartStrategy="drop")
    als_model = als.fit(df_train)
    pred_recs = als_model.recommendForUserSubset(user_subset, 500)
    pred_recs = pred_recs.select('user_idx', F.col('recommendations.track_idx'))

    pred_true_rdd = pred_recs.join(true_recs, on='user_idx').rdd.map(lambda row: (row[1], row[2]))
    metrics = RankingMetrics(pred_true_rdd)
    map_ = metrics.meanAveragePrecision
    ndcg = metrics.ndcgAt(500)
    mpa = metrics.precisionAt(500)
    print('Testing Dataset')
    print('rank: {} reg: {} map: {} ndcg: {} mpa:{}'.format(15, 0.1, map_, ndcg, mpa))
    end = time.time()
    print('elapsed time: ', end - start)


# Only enter this block if we're in main
if __name__ == "__main__":

    # Create the spark session object
    spark = SparkSession.builder.config('spark.sql.autoBroadcastJoinThreshold', '-1').appName('part1').getOrCreate()

    sub_sample = [0.01, 0.05, 0.25, 0.5, 1.0]
    for i in sub_sample:
        # Call our main routine
        main(spark, i)