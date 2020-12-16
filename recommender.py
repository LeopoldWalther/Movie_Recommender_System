import numpy as np
import pandas as pd
import recommender_functions as rf
import sys # can use sys to take command line arguments


class Recommender():
    """
    XXX
    """
    def __init__(self, ):
        """
        Initialization of the Recommender object.
        """


    def fit(self, reviews_path, movies_path, latent_features=15, learning_rate=0.0001, iterations=250):
        """
        This function performs matrix factorization using a basic form of FunkSVD with no regularization

        INPUT:
            reviews_path - (string) path to a matrix with users as rows, movies as columns, and ratings as values
            movies_path - (string) path to a matrx with XXX
            latent_features - (int) the number of latent features used
            learning_rate - (float) the learning rate
            iterations - (int) the number of iterations

        OUTPUT:

        """

        self.reviews = pd.read_csv(reviews_path)
        self.movies = pd.read_csv(movies_path)
        self.latent_features = latent_features
        self.learning_rate = learning_rate
        self.iterations = iterations

        # create user item matrix for collaborative filtering
        user_item = self.reviews[['user_id', 'movie_id', 'rating', 'timestamp']]
        self.user_item_df = user_item.groupby(['user_id', 'movie_id'])['rating'].max().unstack()
        self.user_item_matrix = np.array(self.user_item_df)

        self.amount_users = self.user_item_mat.shape[0]
        self.amount_movies = self.user_item_mat.shape[1]
        self.amount_ratings = np.count_nonzero(~np.isnan(self.user_item_mat))
        self.user_ids_series = np.array(self.user_item_df.index)
        self.movie_ids_series = np.array(self.user_item_df.columns)

        # intialize user and movie matrices with random values for FunkSVD
        user_matrix = np.random.rand(self.amount_users, self.latent_features)
        movie_matrix = np.random.rand(self.latent_features, self.amount_movies)

        # initialize sse at 0 for first iteration
        sum_squared_error_accumulated = 0

        # keep track of iteration and MSE
        print("Optimizaiton Statistics")
        print("Iterations | Mean Squared Error ")

        # for each iteration
        for iteration in range(self.iterations):

            # update our sse
            old_sum_squared_error_accumulated = sum_squared_error_accumulated
            sum_squared_error_accumulated = 0

            # For each user-movie pair
            for i in range(self.amount_users):
                for j in range(self.amount_movies):

                    # if the rating exists
                    if self.user_item_matrix[i, j] > 0:

                        # compute the error as the actual minus the dot product of the user and movie latent features
                        difference = self.user_item_matrix[i, j] - np.dot(user_matrix[i, :], movie_matrix[:, j])

                        # Keep track of the sum of squared errors for the matrix
                        sum_squared_error_accumulated += diff**2

                        # update the values in each matrix in the direction of the gradient
                        for k in range(self.latent_features):
                            user_matrix[i, k] += self.learning_rate * (2 * difference * movie_matrix[k, j])
                            movie_matrix[k, j] += self.learning_rate * (2 * difference * user_matrix[i, k])

            print("%d \t\t %f" % (iteration + 1, sum_squared_error_accumulated / self.amount_ratings))

        # SVD based fit
        self.user_matrix = user_matrix
        self.movie_matrix = movie_matrix

        # Knowledge based fit
        self.ranked_movies = rf.create_ranked_df(self.movies, self.reviews)


    def predict_rating(self, ):
        """
        makes predictions of a rating for a user on a movie-user combo
        """

    def make_recs(self,):
        """
        given a user id or a movie that an individual likes
        make recommendations
        """


if __name__ == '__main__':
    # test different parts to make sure it works
