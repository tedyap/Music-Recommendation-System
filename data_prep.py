from pyspark.sql import SparkSession
from pyspark.ml.feature import StringIndexer
from main import get_data

def main(spark):
    partitions = 1000

    train = get_data(spark, 'cf_train', .01)
    val = get_data(spark, 'cf_validation', .01)
    test = get_data(spark, 'cf_test', .01)

    for column in ['user', 'track']:
        indexer = StringIndexer(inputCol=f'{column}_id', outputCol=f'{column}_idx', handleInvalid='keep')
        indexed = indexer.fit(train)
        train = indexed.transform(train)
        val = indexed.transform(val)
        test = indexed.transform(test)
        indexer.save(f'{column}_indexer')

    train = train.repartition(partitions, 'user_id').select(['user_idx', 'count', 'track_idx'])
    train.write.parquet(path='processed_data/cf_train_idx.parquet', mode='overwrite')

    val = val.repartition(partitions, 'user_id').select(['user_idx', 'count', 'track_idx'])
    val.write.parquet(path='processed_data/cf_validation_idx.parquet', mode='overwrite')

    test = test.repartition(partitions, 'user_id').select(['user_idx', 'count', 'track_idx'])
    test.write.parquet(path='processed_data/cf_test_idx.parquet', mode='overwrite')


if __name__ == "__main__":
    spark = SparkSession.builder.appName('data_prep').getOrCreate()
    main(spark)
