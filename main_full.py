#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Starter Pyspark Script for students to complete for their Lab 3 Assignment.
Usage:
    $ spark-submit main.py <student_netID>
"""

# And pyspark.sql to get the spark session
from pyspark.sql import SparkSession
from pyspark import SparkContext
from pyspark.sql.functions import avg, min, count, desc, countDistinct, asc
from pyspark.ml.feature import StringIndexer
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.mllib.evaluation import RegressionMetrics, RankingMetrics
from pyspark.ml.recommendation import ALS
from pyspark.sql import Row, Column
import pyspark.sql.functions as F
from pyspark import HiveContext
from pyspark.sql.functions import *
from pyspark.sql.window import Window
import pandas as pd

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
    true_label = val.select('user_idx', 'track_idx').groupBy('user_idx').agg(expr('collect_list(track_idx) as true_item'))


    # define paremeter values for parameter tuning
    ranks = [1, 10, 100] # default is 10
    regs = [0.1, 1, 10] # default is 1
    nblocks = [1, 10, 100] # default is 10
    mIters = [1, 10, 100] # default is 10
    alphas = [0.1, 1, 10] # default is 1
    nonnegs = [True, False] # default is False

    count = 0
    best_model = None
    best_map = None
    stats = []
    for rnk in ranks:
        for reg in regs:
            for nblock in nblocks:
                for mIter in mIters:
                    for a in alphas:
                        for nonneg in nonnegs:
                            als = ALS(numBlocks = nblock, rank=rnk, maxIter = mIter, regParam=reg, implicitPrefs = True, alpha = a,  nonnegative = nonneg, userCol="user_idx", itemCol="track_idx", ratingCol="count", coldStartStrategy="drop")
                            model = als.fit(train)
                            user_subset = val.select('user_idx').distinct()
                            userRecs = model.recommendForUserSubset(user_subset, 500)
                            pred_label = userRecs.select('user_idx','recommendations.track_idx')
                            pred_true_rdd = pred_label.join(true_label, 'user_idx', 'inner').select('track_idx','true_item')
                            metrics = RankingMetrics(pred_true_rdd.rdd)
                            MAP = metrics.meanAveragePrecision
                            ndcg = metrics.ndcgAt(500)
                            precision = metrics.precisionAt(500)                            
                            print('map score: {0}, ndcg score: {1}, precision score: {2}, rank: {3}, regParam: {4}, numBlocks: {5}, maxIter: {6}, alpha: {7}, nonnegatives: {8}'.format(MAP, ndcg, precision, rnk, reg, nblock, mIter, a, nonneg))
     break
            
    # best model parameters based on ___ ranking metric: 
    # performance of best model
#     rnk = 
#     reg = 
#     als = ALS(rank=rnk, regParam=reg, userCol="user_idx", itemCol="track_idx", ratingCol="count", coldStartStrategy="drop")
#     model = als.fit(train)
#     user_subset = val.select('user_idx').distinct()
#     userRecs = model.recommendForUserSubset(user_subset, 500)
#     pred_label = userRecs.select('user_idx','recommendations.track_idx')
#     pred_true_rdd = pred_label.join(true_label, 'user_idx', 'inner').select('track_idx','true_item')
#     metrics = RankingMetrics(pred_true_rdd.rdd)
#     map_ = metrics.meanAveragePrecision
#     ndcg = metrics.ndcgAt(500)
#     precision = metrics.precisionAt(500)
#     print('map score: ', map_, 'ndcg score: ', ndcg, 'map score: ', precision)
    
# Only enter this block if we're in main
if __name__ == "__main__":

    # Create the spark session object
    spark = SparkSession.builder.appName('part1').config('spark.blacklist.enabled', False).getOrCreate()
    sc =SparkContext.getOrCreate()

    SUBSET_SIZE = 0.01
    # Call our main routine
    main_full(spark, SUBSET_SIZE)
