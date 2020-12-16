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

        self.amount_users = self.user_item_matrix.shape[0]
        self.amount_movies = self.user_item_matrix.shape[1]
        self.amount_ratings = np.count_nonzero(~np.isnan(self.user_item_matrix))
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
                        sum_squared_error_accumulated += difference ** 2

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


    def predict_rating(self, user_id, movie_id ):
        """
        Makes predictions of a rating for a user on a movie-user combo

        INPUT

        OUTPUT
        """
        try:
            # User row and Movie Column
            user_row = np.where(self.user_ids_series == user_id)[0][0]
            movie_col = np.where(self.movie_ids_series == movie_id)[0][0]

            # Take dot product of that row and column in U and V to make prediction
            pred = np.dot(self.user_matrix[user_row, :], self.movie_matrix[:, movie_col])

            movie_name = str(self.movies[self.movies['movie_id'] == movie_id]['movie']) [5:]
            movie_name = movie_name.replace('\nName: movie, dtype: object', '')
            print("For user {} we predict a {} rating for the movie {}.".format(user_id, round(pred, 2), str(movie_name)))

            return pred

        except:
            print("I'm sorry, but a prediction cannot be made for this user-movie pair.  It looks like one of these items does not exist in our current database.")

            return None


    def make_recommendations(self, _id, _id_type='movie', rec_num=5):
        """
        given a user id or a movie that an individual likes
        make recommendations
        """
        rec_ids, rec_names = None, None

        if _id_type == 'user':
            if _id in self.user_ids_series:
                # Get the index of which row the user is in for use in U matrix
                idx = np.where(self.user_ids_series == _id)[0][0]

                # take the dot product of that row and the V matrix
                preds = np.dot(self.user_matrix[idx,:], self.movie_matrix)

                # pull the top movies according to the prediction
                indices = preds.argsort()[-rec_num:][::-1] #indices
                rec_ids = self.movie_ids_series[indices]
                rec_names = rf.get_movie_names(rec_ids, self.movies)

            else:
                # if we don't have this user, give just top ratings back
                rec_names = rf.popular_recommendations(_id, rec_num, self.ranked_movies)
                print("Because this user wasn't in our database, we are giving back the top movie recommendations for all users. (Cold Start Problem)")

        # Find similar movies if it is a movie that is passed
        else:
            if _id in self.movie_ids_series:
                rec_names = list(rf.find_similar_movies(_id, self.movies))[:rec_num]
            else:
                print("That movie doesn't exist in our database.  Sorry, we don't have any recommendations for you.")

        return rec_ids, rec_names

if __name__ == '__main__':
    # test different parts to make sure it works
    import recommender as r

    #instantiate recommender
    rec = r.Recommender()

    # fit recommender
    rec.fit(reviews_path='data/train_data.csv', movies_path= 'data/movies_clean.csv', learning_rate=.01, iterations= 30)

    # predict
    rec.predict_rating(user_id=8, movie_id=2844)

    # make recommendations
    print(rec.make_recommendations(8,'user')) # user in the dataset
    print(rec.make_recommendations(1,'user')) # user not in dataset
    print(rec.make_recommendations(1853728)) # movie in the dataset
    print(rec.make_recommendations(1)) # movie not in dataset
    print(rec.amount_users)
    print(rec.amount_movies)
    print(rec.amount_ratings)
