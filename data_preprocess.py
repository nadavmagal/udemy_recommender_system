import numpy as np
import pandas as pd
import os
import pickle
import json
from sklearn.utils import shuffle


def main():
    data_full_path = r'/home/nadav/studies/udemy_recommender_sys/archive/rating.csv'
    preprocessed_data_output_dir = r'/home/nadav/studies/udemy_recommender_sys/preprocessed_data'
    n_rows=100000
    raw_data = pd.read_csv(data_full_path, nrows=n_rows)

    movie_index_mapping = {cur_movie_index: ii for ii, cur_movie_index in enumerate(np.unique(list(raw_data['movieId'])))}
    user_index_mapping = {cur_user_index: ii for ii, cur_user_index in enumerate(np.unique(list(raw_data['userId'])))}

    raw_data = shuffle(raw_data)
    cutoff = int(0.8 * len(raw_data))
    raw_data_train = raw_data.iloc[:cutoff]
    raw_data_test = raw_data.iloc[cutoff:]

    users = np.unique(np.array(raw_data['userId']))

    user2movie = dict()
    movie2user = dict()
    usermovie2rating = dict()
    usermovie2rating_test = dict()

    rating_matrix = pd.DataFrame(index=users)

    for index, row in raw_data_train.iterrows():
        add_row_to_dicts(movie2user, row, user2movie, usermovie2rating, movie_index_mapping, user_index_mapping)
        # add_row_to_matrix(rating_matrix, row)
    #add missing movies to movies2user dict
    for ii in range(len(movie_index_mapping)):
        if ii not in movie2user:
            movie2user[ii] = []

    for index, row in raw_data_test.iterrows():
        add_row_to_dicts(None, row, None, usermovie2rating_test, movie_index_mapping, user_index_mapping)

    with open(os.path.join(preprocessed_data_output_dir, f'user2movie_{n_rows}.json'), 'wb') as f:
        pickle.dump(user2movie, f)

    with open(os.path.join(preprocessed_data_output_dir, f'movie2user_{n_rows}.json'), 'wb') as f:
        pickle.dump(movie2user, f)

    with open(os.path.join(preprocessed_data_output_dir, f'usermovie2rating_{n_rows}.json'), 'wb') as f:
        pickle.dump(usermovie2rating, f)

    with open(os.path.join(preprocessed_data_output_dir, f'usermovie2rating_test_{n_rows}.json'), 'wb') as f:
        pickle.dump(usermovie2rating_test, f)

    rating_matrix.to_csv(os.path.join(preprocessed_data_output_dir, f'rating_matrix_{n_rows}.csv'), index=False)

    return rating_matrix, user2movie, movie2user


def add_row_to_dicts(movie2user, row, user2movie, usermovie2rating, movie_index_mapping, user_index_mapping):
    movie_id = movie_index_mapping[row['movieId']]
    user_id = user_index_mapping[row['userId']]
    rating = row['rating']
    if user2movie is not None:
        if user_id not in user2movie:
            user2movie[user_id] = []
        user2movie[user_id].append(movie_id)
    if movie2user is not None:
        if movie_id not in movie2user:
            movie2user[movie_id] = []
        movie2user[movie_id].append(user_id)

    usermovie2rating[(user_id, movie_id)] = rating


def add_row_to_matrix(rating_matrix, row):
    rating = row['rating']
    movie_id = row['movieId']
    user_id = row['userId']
    if movie_id not in rating_matrix.columns:
        rating_matrix[movie_id] = np.nan
    rating_matrix[movie_id][user_id] = rating


if __name__ == "__main__":
    main()
