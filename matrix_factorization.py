import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import pickle
import tqdm

def get_loss(d, W, U, b, c, mu):
    # d: (user_id, movie_id) -> rating
    N = float(len(d))
    sse = 0
    for k, r in d.items():
        i, j = k
        p = W[i].dot(U[j]) + b[i] + c[j] + mu
        sse += (p - r) * (p - r)
    return sse / N


def main():
    preprocessed_data_full_path = r'/home/nadav/studies/udemy_recommender_sys/preprocessed_data'
    n_rows = 100000
    with open(os.path.join(preprocessed_data_full_path, f'user2movie_{n_rows}.json'), 'rb') as f:
        user2movie = pickle.load(f)

    with open(os.path.join(preprocessed_data_full_path, f'movie2user_{n_rows}.json'), 'rb') as f:
        movie2user = pickle.load(f)

    with open(os.path.join(preprocessed_data_full_path, f'usermovie2rating_{n_rows}.json'), 'rb') as f:
        usermovie2rating = pickle.load(f)

    with open(os.path.join(preprocessed_data_full_path, f'usermovie2rating_test_{n_rows}.json'), 'rb') as f:
        usermovie2rating_test = pickle.load(f)

    # rating_matrix = pd.read_csv(os.path.join(preprocessed_data_full_path, f'rating_matrix_{n_rows}.csv'))

    ''' initialize_data '''
    N = len(user2movie)
    M = len(movie2user)
    K = 10

    W = np.random.randn(N, K)
    U = np.random.randn(M, K)

    b = np.zeros(N)
    c = np.zeros(M)
    mu = np.mean(list(usermovie2rating.values()))

    NUM_OF_ITERATION = 80
    reg = 20
    train_losses = []
    test_losses = []
    for cur_iter in tqdm.tqdm(range(NUM_OF_ITERATION)):
        print("iteration:", cur_iter)

        ''' update W and b '''
        for i in range(N):
            # iterating over users
            # initiating cur W
            matrix_for_calc_W = np.eye(K) * reg
            vector_for_calc_W = np.zeros(K)

            # initiating cur b
            bi = 0

            for j in user2movie[i]:
                # iterating over user's movies
                ''' calculate W and b while iterating over movies'''
                r_ij = usermovie2rating[(i, j)]
                matrix_for_calc_W += np.outer(U[j], U[j])
                vector_for_calc_W = (r_ij - b[i] - c[j] - mu) * U[j]
                bi += r_ij - np.dot(W[i], U[j]) - c[j] - mu

            # update W
            W[i] = np.linalg.solve(matrix_for_calc_W, vector_for_calc_W)
            b[i] = bi / (len(user2movie[i]) + reg)

        ''' update U and c '''
        for j in range(M):
            # for U
            matrix = np.eye(K) * reg
            vector = np.zeros(K)

            # for c
            cj = 0
            for i in movie2user[j]:
                r = usermovie2rating[(i, j)]
                matrix += np.outer(W[i], W[i])
                vector += (r - b[i] - c[j] - mu) * W[i]
                cj += (r - W[i].dot(U[j]) - b[i] - mu)

            # set the updates
            U[j] = np.linalg.solve(matrix, vector)
            c[j] = cj / (len(movie2user[j]) + reg)

        train_losses.append(get_loss(usermovie2rating, W, U, b, c, mu))
        test_losses.append(get_loss(usermovie2rating_test, W, U, b, c, mu))
        a = 3

    # plot losses
    plt.figure()
    plt.plot(train_losses, label="train loss")
    plt.plot(test_losses, label="test loss")
    plt.legend()
    plt.show()    return


if __name__ == "__main__":
    main()
