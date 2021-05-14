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
    
    val_users=val.select(['user_idx']).distinct()
    true_label = val.select('user_idx', 'track_idx').groupBy('user_idx').agg(expr('collect_list(track_idx) as true_item'))

    # define paremeter values for parameter tuning
    ranks = [5, 10, 15]
    regs = [0.1, 1, 10]

    count = 0
    best_model = None
    best_map = None
    stats = []
    for rnk in ranks:
        for reg in regs:
            als = ALS(rank=rnk, regParam=reg, userCol="user_idx", itemCol="track_idx", ratingCol="count", coldStartStrategy="drop")
            model = als.fit(train)
            
            
            user_subset = val.select('user_idx').distinct()
            userRecs = model.recommendForUserSubset(user_subset, 500)
            #userRecs = model.recommendForAllUsers(500)
            userRecs=val_users.join(userRecs,'user_idx','inner')
            pred_label = userRecs.select('user_idx','recommendations.track_idx')
            pred_true_rdd = pred_label.join(true_label, 'user_idx', 'inner').select('recommendations.track_idx','true_item')
            metrics = RankingMetrics(sc.parallelize(pred_true_rdd.rdd))
            map_ = metrics.meanAveragePrecision
            ndcg = metrics.ndcgAt(500)
            mpa = metrics.precisionAt(500)
            print('map score: ', map_, 'ndcg score: ', ndcg, 'map score: ', mpa)
            break
        break
            

            
            
            
#             for user in userRecs.select("user_idx").collect():
#                 p = spark.sql("SELECT recommendations.track_idx FROM userRecs WHERE user_idx = "+str(user.user_idx))
#                 predicted = [row.track_idx for row in p.collect()][0]
#                 print(predicted[0]
#                 a = spark.sql("SELECT track_idx FROM val WHERE user_idx = "+str(user.user_idx))
#                 actual = [row.track_idx for row in a.collect()][0]
                
#                 val.filter(user_idx == users[0]).select("track_idx")
            
#             # for testing
#             print(userRecs.count()
#             where(user_recs.user == 0).select("recommendations.item").collect()
#             users = [row.user_idx for row in userRecs.select("user_idx").collect()]
#             print(users[0], users[1], usesr[2], users[3], users[4])
#             p0 = userRecs.filter(userRecs.user_idx == users[0]).select("recommendations")
#             predicted0 = [row.recommendations for row in p.collect()]
#             predicted0_00 = predicted0[0][0]
#             a0 = val.filter(userRecs.user_idx == users[0]).select("track_idx")
#             actual0 = [row.track_idx for row in a.collect()]
            
            
#             predictionAndLabels=[]
#             Counter=0
#             for user in userRecs.select("user_idx").collect():
                
#                 p = userRecs.filter(userRecs.user_idx == user.user_idx).select("recommendations") #should we use .where instead of .filter and "recommendations.track_idx" instead of just "recommendations" (https://spark.apache.org/docs/2.2.0/api/python/pyspark.ml.html#pyspark.ml.recommendation.ALS) 
#                 predicted = [row.recommendations for row in p.collect()][0][0] # shouldn't this be row.recommendations.track_idx
               
#                 a = val.filter(userRecs.user_idx == user.user_idx).select("track_idx")
#                 actual = [row.track_idx for row in a.collect()]

#                 predictionAndLabels.append((predicted,actual))
                
#                 if Counter==0:
#                     print(user)
#                     print("predicted is:",predicted)
#                     print("actual is:",actual)
                
#                 Counter+=1
                
#             predictionAndLabels = sc.parallelize(predictionAndLabels)
#             metrics = RankingMetrics(predictionAndLabels)
#             MAP = metrics.meanAveragePrecision
#             NDCG=metrics.ndcgAt(500)
#             PAT=metrics.precisionAt(500)
#             print("Rank is:{}, Reg is:{},MAP is:{},NDCG is:{}, PAT is:{}".format(rnk,reg,MAP,NDCG,PAT))
            
            
        
            
            
            
#             print(predictions)
            
#             metrics_df=predictions.select(['prediction_rank','count_rank'])
#             metrics = RankingMetrics(metrics_df.rdd)
#             MAP=metrics.meanAveragePrecision
            
#             #iterate over each row in predictions dataframe
#             #append ()
            
#             evaluator = RegressionEvaluator(metricName="rmse", labelCol="count_rank", predictionCol="prediction_rank")
#             rmse = evaluator.evaluate(predictions)
            
#             print('Current model: Rank:'+str(rnk)+', RegParam: '+str(reg)+', RMSE: '+str(rmse)+"MAP: "+str(MAP))

#             #userRecs = model.recommendForAllUsers(500).show(5)

#             if count == 0:
#                 best_model = model
#                 best_map = MAP
#                 stats = [rnk, reg, rmse,MAP]
#                 count += 1
#             else:
#                 if MAP < best_map:
#                     best_model = model
#                     best_map = MAP
#                     stats = [rnk, reg, rmse, MAP]
#     print('Best model: Rank: {}, RegParam: {}, RMSE: {}, MAP: {}'.format(*stats))
    
#     rnk=stats[0]
#     reg=stats[1]
    
#     als = ALS(rank=rnk, regParam=reg, userCol="user_idx", itemCol="track_idx", ratingCol="count", implicitPrefs=True, coldStartStrategy="drop")
#     model = als.fit(train)
#     predictions = model.transform(test)
    
#     predictions=predictions.withColumn("count_rank", rank().over(Window.partitionBy("user_idx").orderBy(desc("count"))))
#     predictions=predictions.withColumn("prediction_rank", rank().over(Window.partitionBy("user_idx").orderBy(desc("prediction"))))
#     predictions=predictions.filter(predictions.prediction_rank<=500)
            
#     metrics_df=predictions.select(['prediction_rank','count_rank'])
#     metrics = RankingMetrics(metrics_df.rdd)
#     MAP=metrics.meanAveragePrecision
    
#     evaluator = RegressionEvaluator(metricName="rmse", labelCol="count_rank", predictionCol="prediction_rank")
#     rmse = evaluator.evaluate(predictions)
    
#     print('Test Set: Rank:'+str(rnk)+', RegParam: '+str(reg)+', RMSE: '+str(rmse)+"MAP: "+str(MAP))
    
# Only enter this block if we're in main
if __name__ == "__main__":

    # Create the spark session object
    spark = SparkSession.builder.appName('part1').config('spark.blacklist.enabled', False).getOrCreate()
    sc =SparkContext.getOrCreate()

    SUBSET_SIZE = 0.01
    # Call our main routine
    main_full(spark, SUBSET_SIZE)
