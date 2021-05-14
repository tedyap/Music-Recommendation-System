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

def main_full(SUBSET_SIZE):
    train = get_data('cf_train_new', SUBSET_SIZE)
    val = get_data('cf_validation', SUBSET_SIZE)
    test = get_data('cf_test', SUBSET_SIZE)
    damps = [.25] #.5, 1, 2, 5, 10, 15, 30, 50, 100, 150]
    items = train['item'].tolist()
    items = list(set(items))

    with open(f'/scratch/sk8520/temp/final-project-if_it_works_dont_touch_it/output.txt', mode='w') as f:
        f.write(f'{damps}\n')
        f.write('file sizes: \n')
        f.write(f'train: {len(train)}\n')
        f.write(f'val: {len(val)}\n')
        f.write(f'test: {len(test)}\n')


        ## predict rating for each user, item pair
        # for damp in damps:
        #     print(damp)
        #     b = bias.Bias(items=True, users=True, damping=damp).fit(train)
        #     preds = [b.predict_for_user(user=row['user'], items=[row['item']]).values[0] for index, row in val.iterrows()]
        #     true_preds = val['rating'].tolist()
        #     rmse = mean_squared_error(y_true=true_preds, y_pred=preds)
        #     f.write(f'damping parameter: {damp} mean squared error: {rmse}\n')



        ## predict top item for each user
        true_preds = val['item'].tolist()
        for damp in damps:
            print(damp)
            b = bias.Bias(items=True, users=True, damping=damp).fit(train)
            preds = []
            for index, row in val.iterrows():
                pred = b.predict_for_user(user=row['user'], items=items).values
                max_item_position = np.argmax(pred)
                preds.append(items[max_item_position])
            score = accuracy_score(preds, true_preds)
            print(score)



if __name__ == "__main__":

    SUBSET_SIZE = .005
    # Call our main routine
    main_full(SUBSET_SIZE)
