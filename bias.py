#!/usr/bin/env python
# -*- coding: utf-8 -*-


from lenskit.algorithms import bias
import pandas as pd
from sklearn.metrics import average_precision_score

def get_data(file_name, frac_keep):
    df = pd.read_parquet(f'/scratch/work/courses/DSGA1004-2021/MSD/{file_name}.parquet')
    print(df.shape)
    df = df.sample(replace=False, frac=frac_keep, random_state=1)
    df.rename(columns={'count':'rating', 'track_id':'item', 'user_id':'user'}, inplace=True)
    return df

def main_full(SUBSET_SIZE):
    train = get_data('cf_train_new', SUBSET_SIZE)
    val = get_data('cf_validation', SUBSET_SIZE)
    test = get_data('cf_test', SUBSET_SIZE)


    for damp in [.5, 1, 2, 5]:
        b = bias.Bias(items=True, users=True, damping=damp).fit(train)
        preds = [b.predict_for_user(user=row['user'], items=[row['item']]).values[0] for index, row in val.iterrows()]
        true_preds = val['rating'].tolist()
        map = average_precision_score(y_true=true_preds, y_score=preds)
        print(damp, map)

if __name__ == "__main__":

    SUBSET_SIZE = .01
    # Call our main routine
    main_full(SUBSET_SIZE)
