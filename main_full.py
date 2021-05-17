#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Starter Pyspark Script for students to complete for their Lab 3 Assignment.
Usage:
    $ spark-submit main.py <student_netID>
"""

# And pyspark.sql to get the spark session
from pyspark import SparkContext
from pyspark.sql.functions import avg, min, count, desc, countDistinct, asc
from pyspark.sql import Row, Column
from pyspark import HiveContext
from pyspark.sql.functions import *
from pyspark.sql.window import Window
from pyspark.sql import SparkSession
from pyspark import SparkConf
from pyspark.ml.feature import StringIndexer
from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.mllib.evaluation import RankingMetrics
import pyspark.sql.functions as F
from pyspark.sql.functions import expr
import itertools as it
import random
import numpy as np
import pandas as pd

def get_data(spark, file_name, frac_keep):
    # function to read and sample from dataset with constant seed across datasets
    df = spark.read.parquet(f'hdfs:/user/bm106/pub/MSD/{file_name}.parquet')
    if frac_keep<1:
        df = df.sample(False, frac_keep, 1)
    return df


def main_full(spark,SUBSET_SIZE):
    '''Main routine for Final Project
    Parameters
    ----------
    spark : SparkSession object
    '''
    # load and sample from datasets
    
    train = get_data(spark, 'cf_train_new', SUBSET_SIZE)
    val = get_data(spark, 'cf_validation', SUBSET_SIZE)
    test = get_data(spark, 'cf_test', SUBSET_SIZE)

    # StringIndexing
    for column in ['user', 'track']:
        indexer = StringIndexer(inputCol=f'{column}_id', outputCol=f'{column}_idx', handleInvalid='keep')
        indexed = indexer.fit(train)
        train = indexed.transform(train)
        val = indexed.transform(val)
        test = indexed.transform(test)
        indexer.write().overwrite().save(f'{column}_indexer')

    train = train.select(['user_idx', 'count', 'track_idx'])
    val = val.select(['user_idx', 'count', 'track_idx'])
    test = test.select(['user_idx', 'count', 'track_idx'])
    true_label_val = val.select('user_idx', 'track_idx').groupBy('user_idx').agg(expr('collect_list(track_idx) as true_item'))
    true_label_test = test.select('user_idx', 'track_idx').groupBy('user_idx').agg(expr('collect_list(track_idx) as true_item'))

    # define paremeter values for parameter tuning
    ranks = [1, 10, 100]
    regs = [0.1, 1, 10]

    count = 0
    best_model = None
    best_map = None
    stats = []
    for rnk in ranks:
        for reg in regs:
            als = ALS(rank=rnk, regParam=reg, userCol="user_idx", itemCol="track_idx", ratingCol="count", coldStartStrategy="drop",implicitPrefs=True)
            model = als.fit(train)
            
            # validation
            user_subset_val = val.select('user_idx').distinct()
            userRecs_val = model.recommendForUserSubset(user_subset_val, 500)
            pred_label_val = userRecs_val.select('user_idx','recommendations.track_idx')
            pred_true_rdd_val = pred_label_val.join(true_label_val, 'user_idx', 'inner').select('track_idx','true_item')
            metrics_val = RankingMetrics(pred_true_rdd_val.rdd)
            map_val = metrics_val.meanAveragePrecision
            ndcg_val = metrics_val.ndcgAt(500)
            precision_val = metrics_val.precisionAt(500)
            
            
            # test
            user_subset_test = test.select('user_idx').distinct()
            userRecs_test = model.recommendForUserSubset(user_subset_test, 500)
            pred_label_test = userRecs_test.select('user_idx','recommendations.track_idx')
            pred_true_rdd_test = pred_label_test.join(true_label_test, 'user_idx', 'inner').select('track_idx','true_item')
            metrics_test = RankingMetrics(pred_true_rdd_test.rdd)
            map_test = metrics_test.meanAveragePrecision
            ndcg_test = metrics_test.ndcgAt(500)
            precision_test = metrics_test.precisionAt(500)
            
            # results
            print('VALIDATION SCORES val map score: ', map_val, ' val ndcg score: ', ndcg_val, ' val precision score: ', precision_val, 
                  ' PARAMETERS rank: ', rnk, ' regParam: ', reg, ' TEST SCORES test map score: ', map_test,' test ndcg score: ', 
                  ndcg_test, ' val precision score: ', precision_test)
           

    
# Only enter this block if we're in main
if __name__ == "__main__":

    # Create the spark session object
    spark = SparkSession.builder.appName('part1').config('spark.blacklist.enabled', False).getOrCreate()
    sc =SparkContext.getOrCreate()

    SUBSET_SIZE = 1
    # Call our main routine
    main_full(spark, SUBSET_SIZE)
