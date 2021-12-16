import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import copy


def create_rating_matrix(raw_data):
    users = np.unique(np.array(raw_data['userId']))
    rating_matrix = pd.DataFrame(index=users)
    for index, row in raw_data.iterrows():
        movie_id = row['movieId']
        user_id = row['userId']
        rating = row['rating']
        if movie_id not in rating_matrix.columns:
            rating_matrix[movie_id] = np.nan
        rating_matrix[movie_id][user_id] = rating

    return rating_matrix


def split_test_train_data(rating_matrix, train_test_ratio):
    split_index = int(len(rating_matrix[rating_matrix.columns[0]]) // (1 / train_test_ratio))
    train = rating_matrix.iloc[split_index:, :]
    test = rating_matrix.iloc[:split_index, :]
    return train, test


def calculate_user_bias(rating_matrix):
    user_bias = rating_matrix.mean(axis=1)
    return user_bias


def get_user_weights_pearson_correlation(test_rating_matrix_wo_cur_test, test_user_id, users_bias):
    w_i_itag = pd.DataFrame(index=test_rating_matrix_wo_cur_test.index)
    w_i_itag[0] = np.nan
    for cur_user_id in test_rating_matrix_wo_cur_test.index:
        '''
        i - user to test
        i_tag - iterative user
        '''
        user_i_rating_matrix = np.array(test_rating_matrix_wo_cur_test.loc[test_user_id])
        user_itag_rating_matrix = np.array(test_rating_matrix_wo_cur_test.loc[cur_user_id])

        user_i_bias = users_bias.loc[test_user_id]
        user_itag_bias = users_bias.loc[cur_user_id]

        # find shared movies group
        shared_movies_indices = np.where(
            np.bitwise_and(~np.isnan(user_i_rating_matrix), ~np.isnan(user_itag_rating_matrix)))
        user_i_rating_matrix_shared = user_i_rating_matrix[shared_movies_indices]
        user_itag_rating_matrix_shared = user_itag_rating_matrix[shared_movies_indices]

        w_i_itag_numerator = np.sum(
            np.multiply(user_i_rating_matrix_shared - user_i_bias, user_itag_rating_matrix_shared - user_itag_bias))
        w_i_itag_denominator = np.sqrt(np.sum(np.square(user_i_rating_matrix_shared - user_i_bias))) * np.sqrt(
            np.sum(np.square(user_itag_rating_matrix_shared - user_itag_bias)))

        cur_w_i_itag = w_i_itag_numerator / w_i_itag_denominator
        w_i_itag.loc[cur_user_id, 0] = cur_w_i_itag
    return w_i_itag

def get_estimated_rating(test_movie_id, test_rating_matrix_wo_cur_test, test_user_id, users_bias, wii_tag):
    participated_user_indices = np.array(
        np.where(~np.isnan(np.array(test_rating_matrix_wo_cur_test[test_movie_id])))[0])
    if len(participated_user_indices)==0:
        return np.nan
    participated_rating = np.array(test_rating_matrix_wo_cur_test[test_movie_id])[participated_user_indices]
    participated_weights = np.squeeze(np.array(wii_tag))[participated_user_indices]
    participated_bias = np.array(users_bias)[participated_user_indices]
    estimated_test_user_score = users_bias.loc[test_user_id] + np.sum(
        np.multiply(participated_weights, participated_rating - participated_bias)) / np.sum(
        np.abs(participated_weights))
    return estimated_test_user_score

def main():
    data_full_path = r'/home/nadav/studies/udemy_recommender_sys/archive/rating.csv'
    raw_data = pd.read_csv(data_full_path, nrows=1000)

    rating_matrix = create_rating_matrix(raw_data)

    test_values_indices = np.array(np.where(np.logical_not(np.isnan(rating_matrix)))).T

    gt_scores_values = []
    estimated_rating_values = []

    for ii, cur_test_indices in enumerate(test_values_indices):
        if ii > 100:
            break
        test_rating_matrix_wo_cur_test = copy.deepcopy(rating_matrix)
        cur_test_rating_gt = test_rating_matrix_wo_cur_test.iloc[cur_test_indices[0], cur_test_indices[1]]
        test_rating_matrix_wo_cur_test.iloc[cur_test_indices[0], cur_test_indices[1]] = np.nan

        test_user_id = test_rating_matrix_wo_cur_test.index[cur_test_indices[0]]
        test_movie_id = test_rating_matrix_wo_cur_test.columns[cur_test_indices[1]]

        users_bias = calculate_user_bias(test_rating_matrix_wo_cur_test)

        wii_tag = get_user_weights_pearson_correlation(test_rating_matrix_wo_cur_test, test_user_id, users_bias)

        a=3

        estimated_test_user_score = get_estimated_rating(test_movie_id, test_rating_matrix_wo_cur_test, test_user_id,
                                                         users_bias, wii_tag)

        gt_scores_values.append(cur_test_rating_gt)
        estimated_rating_values.append(estimated_test_user_score)
    gt_scores_values = np.array(gt_scores_values)
    estimated_rating_values = np.array(estimated_rating_values)
    mse = np.nanmean(np.square(gt_scores_values-estimated_rating_values))
    both = np.vstack([gt_scores_values, estimated_rating_values])
    a=3





if __name__ == "__main__":
    main()
