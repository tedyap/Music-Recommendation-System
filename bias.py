#!/usr/bin/env python
# -*- coding: utf-8 -*-


from lenskit.algorithms import bias
import pandas as pd

def get_data(file_name, frac_keep):
    df = pd.read_parquet(f'/scratch/work/courses/DSGA1004-2021/MSD/{file_name}.parquet')
    df = df.sample(replace=False, frac=frac_keep, random_state=1)
    df.rename(columns={'count':'rating', 'track_id':'item', 'user_id':'user'}, inplace=True)
    return df


def main_full(SUBSET_SIZE):
    train = get_data('cf_train_new', SUBSET_SIZE)
    val = get_data('cf_validation', SUBSET_SIZE)
    test = get_data('cf_test', SUBSET_SIZE)

    b = bias.Bias(items=True, users=True, damping=0).fit(train)
    for index, row in val.iterrows():
        print(index, row)
        break
    preds = [b.predict_for_user(user=row['user'], items=row['item']) for index, row in val.iterrows()]
    print(len(preds))

if __name__ == "__main__":

    SUBSET_SIZE = .01
    # Call our main routine
    main_full(SUBSET_SIZE)
