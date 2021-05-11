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
from pyspark.sql import Row

from pyspark.sql.functions import *
from pyspark.sql.window import Window
import pandas as pd

def get_data(spark, file_name, frac_keep):
    # function to read and sample from dataset with constant seed across datasets
    df = spark.read.parquet(f'hdfs:/user/bm106/pub/MSD/{file_name}.parquet')
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


    # define paremeter values for parameter tuning
    ranks = [5]#[5, 10, 15]
    regs = [0.1]#[0.1, 1, 10]

    count = 0
    best_model = None
    best_map = None
    stats = []
    for rnk in ranks:
        for reg in regs:
            als = ALS(rank=rnk, regParam=reg, userCol="user_idx", itemCol="track_idx", ratingCol="count", implicitPrefs=True, coldStartStrategy="drop")
            model = als.fit(train)
            predictions = model.transform(val)
           
            predictions=predictions.withColumn("count_rank", rank().over(Window.partitionBy("user_idx").orderBy(desc("count"))))
            predictions=predictions.withColumn("prediction_rank", rank().over(Window.partitionBy("user_idx").orderBy(desc("prediction"))))
           
            #predictions_rmse=predictions.filter(predictions.prediction_rank<=500)
            #evaluator = RegressionEvaluator(metricName="rmse", labelCol="count_rank", predictionCol="prediction_rank")
            #rmse = evaluator.evaluate(predictions_rmse)
            
            p = predictions.toPandas()
            p_count=p.sort_values(['user_idx','count_rank'],False)
            p_prediction=p.sort_values(['user_idx','prediction_rank'],False)
            
            users=p['user_idx'].unique().tolist()
            
            predictionAndLabels=[]
            for user in users:
                count_items=p_count[p_count['user_idx']==user]["track_idx"].head(500)
                prediction_items=p_prediction[p_prediction['user_idx']==user]["track_idx"].head(500)
                predictionAndLabels.append((prediction_items,count_items))
               
            predictionAndLabels = sc.parallelize(predictionAndLabels)
            metrics = RankingMetrics(predictionAndLabels)
            MAP = metrics.meanAveragePrecision
            
            print(MAP)
            
            break
        break

            
            
            
            
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

    SUBSET_SIZE = .01
    # Call our main routine
    main_full(spark, SUBSET_SIZE)
