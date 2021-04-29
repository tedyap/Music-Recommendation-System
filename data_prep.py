"""
Spark job to perform string indexing
Usage:
    $ spark-submit data_prep.py
"""

from pyspark.sql import SparkSession
from pyspark.ml.feature import StringIndexer
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS
from pyspark.sql import Row

def main(spark):
    train = spark.read.parquet(f'hdfs:/user/bm106/pub/MSD/cf_train.parquet')
    val = spark.read.parquet(f'hdfs:/user/bm106/pub/MSD/cf_validation.parquet')
    test = spark.read.parquet(f'hdfs:/user/bm106/pub/MSD/cf_test.parquet')

    for column in ['user', 'track']:
        indexer = StringIndexer(inputCol=f'{column}_id', outputCol=f'{column}_idx', handleInvalid='keep')
        indexed = indexer.fit(train)
        train = indexed.transform(train)
        val = indexed.transform(val)
        test = indexed.transform(test)
        indexer.write().overwrite().save(f'{column}_indexer')

    train = train.select(['user_idx', 'count', 'track_idx'])
    train.write.parquet(path='processed_data/cf_train_idx.parquet', mode='overwrite')
    train.unpersist()

    val = val.select(['user_idx', 'count', 'track_idx'])
    val.write.parquet(path='processed_data/cf_validation_idx.parquet', mode='overwrite')
    val.unpersist()

    test = test.select(['user_idx', 'count', 'track_idx'])
    test.write.parquet(path='processed_data/cf_test_idx.parquet', mode='overwrite')
    test.unpersist()
    
def ALS(training,test,regularizationparam,userCol,itemCol,ratingCol):
    #create and fit model, function args control type of fit
    als = ALS(maxIter=5, regParam=regularizationparam, userCol=userCol, itemCol=itemCol, ratingCol=ratingCol, coldStartStrategy="drop")
    model = als.fit(training)
    predictions = model.transform(test)
    evaluator = RegressionEvaluator(metricName="rmse", labelCol="rating", predictionCol="prediction")
    rmse = evaluator.evaluate(predictions)
    return RMSE

if __name__ == "__main__":
    spark = SparkSession.builder.appName('data_prep').getOrCreate()
    main(spark)
