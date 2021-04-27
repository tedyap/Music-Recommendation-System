#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''Starter Pyspark Script for students to complete for their Lab 3 Assignment.
Usage:
    $ spark-submit main.py <student_netID>
'''
#Use getpass to obtain user netID
import getpass

# And pyspark.sql to get the spark session
from pyspark.sql import SparkSession
from pyspark.sql.functions import avg, min, count, desc, countDistinct, asc


def get_data(spark, netID, file_name, frac_keep):
    df = spark.read.parquet(f'hdfs:/user/bm106/pub/MSD/{file_name}.parquet')
    df = df.sample(False, frac_keep, 1)
    return df

def main(spark, netID, SUBSET_SIZE):
    '''Main routine for Final Project
    Parameters
    ----------
    spark : SparkSession object
    netID : string, netID of student to find files in HDFS
    '''
    print('Loading in files')

    df_train = get_data(spark, netID, 'cf_train', SUBSET_SIZE)
    print((df_train.count(), len(df_train.columns)))

    df_val = get_data(spark, netID, 'cf_validation', SUBSET_SIZE)
    df_test = get_data(spark, netID, 'cf_test', SUBSET_SIZE)


# Only enter this block if we're in main
if __name__ == "__main__":

    # Create the spark session object
    spark = SparkSession.builder.appName('part1').getOrCreate()

    # Get user netID from the command line
    netID = getpass.getuser()

    SUBSET_SIZE = .01
    # Call our main routine
    main(spark, netID, SUBSET_SIZE)
