#!/usr/bin/env python
# -*- coding: utf-8 -*-


from lenskit.algorithms import bias
import pandas as pd
from sklearn.metrics import mean_squared_error, average_precision_score, accuracy_score
import numpy as np

def get_data(file_name, frac_keep):
    df = pd.read_parquet(f'/scratch/work/courses/DSGA1004-2021/MSD/{file_name}.parquet')
    df = df.sample(replace=False, frac=frac_keep, random_state=1)
    df.rename(columns={'count':'rating', 'track_id':'item', 'user_id':'user'}, inplace=True)
    return df


def map_score(preds, truth):
    num_positive, num_total = 0, 0
    final_score = 0
    truth = set(truth)

    for item in preds:
        num_total += 1
        if item in truth:
            num_positive += 1
            final_score += num_positive/num_total
    final_score *= num_positive/num_total
    return final_score


def main_full(SUBSET_SIZE):
    train = get_data('cf_train_new', SUBSET_SIZE)
    val = get_data('cf_validation', SUBSET_SIZE)
    test = get_data('cf_test', SUBSET_SIZE)
    damps = [.25]#, .5, 1, 2, 5, 10, 15, 30, 50, 100, 150]

    unique_items = train['item'].unique()

    gb = val.groupby(['user'])
    result = gb['item'].unique()
    # has a list of tracks listened to per each user
    result = result.reset_index()

    with open(f'/scratch/sk8520/temp/final-project-if_it_works_dont_touch_it/output.txt', mode='w') as f:
        f.write(f'{damps}\n')
        f.write('file sizes: \n')
        f.write(f'train: {len(train)}\n')
        f.write(f'val: {len(val)}\n')
        f.write(f'test: {len(test)}\n')

        for damp in damps:
            print(f'Computing {damp}')
            bias_model = bias.Bias(items=True, users=True, damping=damp).fit(train)
            rating_bias = bias_model.transform(train)
            average_utility = rating_bias.groupby('item')['rating'].count()
            top500 = average_utility.nlargest(n=500)
            top500 = top500.index.values.tolist()
            scores = [map_score(top500, x) for x in result['item']]
            print(f'Mean average precision for damping: {damp}: {sum(scores)/len(scores)}')
            f.write(f'Mean average precision for damping: {damp}: {sum(scores)/len(scores)}\n')

            scores = [map_score(bias_model.predict_for_user(user=row['user'], items=unique_items).nlargest(n=500), row['item']) for index, row in result.iterrows()]
            print(f'Mean average precision for damping using predictions: {damp}: {sum(scores)/len(scores)}\n')
            f.write(f'Mean average precision for damping: {damp}: {sum(scores)/len(scores)}\n')



if __name__ == "__main__":

    SUBSET_SIZE = .05
    # Call our main routine
    main_full(SUBSET_SIZE)













# true_label = val.select('user_idx', 'track_idx').groupBy('user_idx').agg(expr('collect_list(track_idx) as true_item'))
# # list of tracks for each user, all positive and unsorted. could be a list of 50 or something
#
#
# # dataframe of user, with list of top 500
# userRecs = model.recommendForUserSubset(user_subset, 500)
