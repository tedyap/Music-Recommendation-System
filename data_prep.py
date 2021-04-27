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

    train_data = get_data(spark, 'cf_train', .01)
    validation_data = get_data(spark, 'cf_validation', .01)
    test_data = get_data(spark, 'cf_test', .01)

    # train_len = train.count()
    # val_len = val.count()
    # test_len = test.count()
    #
    # print('Begin indexing...')
    #
    # for column in ['user', 'track']:
    #     indexer = StringIndexer(inputCol=f'{column}_id', outputCol=f'{column}_idx', handleInvalid='keep')
    #     indexed = indexer.fit(train)
    #     train = indexed.transform(train)
    #     val = indexed.transform(val)
    #     test = indexed.transform(test)
    #     indexer.write().overwrite().save(f'{column}_indexer')
    #
    # print('Finished indexing...')

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

    val_test_data = test_data.union(validation_data)
    train_optional_user = train_data.join(val_test_data, "user_id", "left_anti")
    train_data = train_data.join(train_optional_user, "user_id", "left_anti")

    uid_indexer = StringIndexer(inputCol="user_id", outputCol="user_num", handleInvalid="skip")
    tid_indexer = StringIndexer(inputCol="track_id", outputCol="track_num", handleInvalid="skip")
    model_uid = uid_indexer.fit(train_data)
    model_tid = tid_indexer.fit(train_data)

    uid_train_index = model_uid.transform(train_data)
    combo_train_index = model_tid.transform(uid_train_index)

    uid_val_index = model_uid.transform(validation_data)
    combo_val_index = model_tid.transform(uid_val_index)

    uid_test_index = model_uid.transform(test_data)
    combo_test_index = model_tid.transform(uid_test_index)

    model_uid.save('model_uid')
    model_tid.save('model_tid')

    combo_train_index = combo_train_index.repartition(partitions, "user_id")
    combo_val_index = combo_val_index.repartition(partitions, "user_id")
    combo_test_index = combo_test_index.repartition(partitions, "user_id")

    combo_train_index = combo_train_index.select(["user_num", "count", "track_num"])
    combo_train_index.write.parquet(path='train_index.parquet', mode='overwrite')
    combo_train_index.unpersist()

    combo_val_index = combo_val_index.select(["user_num", "count", "track_num"])
    combo_val_index.write.parquet(path='val_index.parquet', mode='overwrite')
    combo_val_index.unpersist()

    combo_test_index = combo_test_index.select(["user_num", "count", "track_num"])
    combo_test_index.write.parquet(path='test_index.parquet', mode='overwrite')
    combo_test_index.unpersist()


if __name__ == "__main__":
    spark = SparkSession.builder.appName('data_prep').getOrCreate()
    main(spark)
