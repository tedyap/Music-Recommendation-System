from pyspark.sql import SparkSession
from pyspark.ml.feature import StringIndexer
from main import get_data

def main(spark):
    partitions = 1000

    train = get_data(spark, 'cf_train', .01)
    val = get_data(spark, 'cf_validation', .01)
    test = get_data(spark, 'cf_test', .01)

    train_len = train.count()
    val_len = val.count()
    test_len = test.count()

    print('Begin indexing...')

    for column in ['user', 'track']:
        indexer = StringIndexer(inputCol=f'{column}_id', outputCol=f'{column}_idx', handleInvalid='keep')
        indexed = indexer.fit(train)
        train = indexed.transform(train)
        val = indexed.transform(val)
        test = indexed.transform(test)
        indexer.write().overwrite().save(f'{column}_indexer')

    print('Finished indexing...')

    # train = train.repartition(partitions, 'user_id').select(['user_idx', 'count', 'track_idx'])
    # assert train_len == train.count()
    # train.write.parquet(path='processed_data/cf_train_idx.parquet', mode='overwrite')
    # train.unpersist()
    #
    # val = val.repartition(partitions, 'user_id').select(['user_idx', 'count', 'track_idx'])
    # assert val_len == val.count()
    # val.write.parquet(path='processed_data/cf_validation_idx.parquet', mode='overwrite')
    # val.unpersist()
    #
    # test = test.repartition(partitions, 'user_id').select(['user_idx', 'count', 'track_idx'])
    # assert test_len == test.count()
    # test.write.parquet(path='processed_data/cf_test_idx.parquet', mode='overwrite')
    # test.unpersist()


if __name__ == "__main__":
    spark = SparkSession.builder.appName('data_prep').getOrCreate()
    main(spark)
