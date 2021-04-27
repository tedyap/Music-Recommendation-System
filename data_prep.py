"""
Spark job to perform string indexing
Usage:
    $ spark-submit --py-files main.py data_prep.py
"""

from pyspark.sql import SparkSession
from pyspark.ml.feature import StringIndexer
from main import get_data


def main(spark):
    partitions = 1000

    train = get_data(spark, 'cf_train', 1.0)
    val = get_data(spark, 'cf_validation', 1.0)
    test = get_data(spark, 'cf_test', 1.0)

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
