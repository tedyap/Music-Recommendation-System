"""
Spark job to perform string indexing
Usage:
    $ spark-submit --py-files main.py data_prep.py
"""

from pyspark.sql import SparkSession
from pyspark.ml.feature import StringIndexer


def main(spark):
    train = spark.read.parquet(f'hdfs:/user/bm106/pub/MSD/cf_train.parquet')
    val = spark.read.parquet(f'hdfs:/user/bm106/pub/MSD/cf_validation.parquet')
    test = spark.read.parquet(f'hdfs:/user/bm106/pub/MSD/cf_test.parquet')

    print(train.count())
    print('Begin indexing...')

    for column in ['user', 'track']:
        indexer = StringIndexer(inputCol=f'{column}_id', outputCol=f'{column}_idx', handleInvalid='keep')
        indexed = indexer.fit(train)
        train = indexed.transform(train)
        val = indexed.transform(val)
        test = indexed.transform(test)
        indexer.write().overwrite().save(f'{column}_indexer')

    print('Finished indexing...')

    train = train.select(['user_idx', 'count', 'track_idx'])
    print(train.count())
    train.write.parquet(path='processed_data/cf_train_idx.parquet', mode='overwrite')
    train.unpersist()

    val = val.select(['user_idx', 'count', 'track_idx'])
    val.write.parquet(path='processed_data/cf_validation_idx.parquet', mode='overwrite')
    val.unpersist()

    test = test.select(['user_idx', 'count', 'track_idx'])
    test.write.parquet(path='processed_data/cf_test_idx.parquet', mode='overwrite')
    test.unpersist()


if __name__ == "__main__":
    spark = SparkSession.builder.appName('data_prep').getOrCreate()
    main(spark)
