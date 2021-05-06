#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Starter Pyspark Script for students to complete for their Lab 3 Assignment.
Usage:
    $ spark-submit main.py <student_netID>
"""


# And pyspark.sql to get the spark session
from pyspark.sql import SparkSession
from pyspark.sql.functions import avg, min, count, desc, countDistinct, asc
from pyspark.ml.feature import StringIndexer
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS
from pyspark.sql import Row


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
    best_rmse = None
    stats = []
    for rank in ranks:
        for reg in regs:
            als = ALS(rank=rank, regParam=reg, userCol="user_idx", itemCol="track_idx", ratingCol="count", implicitPrefs=True, coldStartStrategy="drop")
            model = als.fit(train)
           
        
            predictions = model.transform(val)
            predictions.show(10)
            evaluator = RegressionEvaluator(metricName="rmse", labelCol="count", predictionCol="prediction")
            rmse = evaluator.evaluate(predictions)
    
            
            print('Current model: Rank:'+str(rank)+', RegParam: '+str(reg)+', RMSE: '+str(rmse))
            
            userRecs = model.recommendForAllUsers(10)
            
            if count == 0:
                best_model = model
                best_rmse = rmse
                stats = [rank, reg, rmse]
                count += 1
            else:
                if rsme < best_rmse:
                    best_model = model
                    best_rmse = rmse
                    stats = [rank, reg, rmse]
    print('Best model: Rank: {}, RegParam: {}, RMSE: {}'.format(*stats))

# Only enter this block if we're in main
if __name__ == "__main__":

    # Create the spark session object
    spark = SparkSession.builder.appName('part1').config('spark.blacklist.enabled', False).getOrCreate()

    SUBSET_SIZE = .01
    # Call our main routine
    main_full(spark, SUBSET_SIZE)
