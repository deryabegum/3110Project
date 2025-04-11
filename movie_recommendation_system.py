
#Movie Recommendation System
# Derya Ari, Burak Cakir, Selim Maral

import os
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
import requests
from io import BytesIO
from zipfile import ZipFile
import warnings
warnings.filterwarnings('ignore')

class MovieRecommendationSystem:
    #the main class for the movie recommendation system"
    
    def __init__(self, data_path=None):
        #initializing the recommendation system
        self.movies_df = None
        self.ratings_df = None
        self.users_df = None
        self.movie_features = None
        self.similarity_matrix = None
        self.combined_features = None
        self.user_preferences = []
        
        #load the data or download it
        if data_path:
            self.load_data(data_path)
        else:
            self.download_and_load_data()
            
        #preperation of data
        self.preprocess_data()
        self.create_movie_features()
    
    def download_and_load_data(self):
        #download the MovieLens 100K dataset & load it
        print("Downloading MovieLens 100K dataset")
        url = "https://files.grouplens.org/datasets/movielens/ml-100k.zip"
        response = requests.get(url)
        
        with ZipFile(BytesIO(response.content)) as zip_ref:
            #extracting it to a temporary directory
            temp_dir = "temp_dataset"
            os.makedirs(temp_dir, exist_ok=True)
            zip_ref.extractall(temp_dir)
            
            #loading the datasets
            self.movies_df = pd.read_csv(
                f"{temp_dir}/ml-100k/u.item", 
                sep='|', 
                encoding='latin-1',
                header=None,
                names=['movie_id', 'title', 'release_date', 'video_release_date', 'IMDb_URL', 
                      'unknown', 'Action', 'Adventure', 'Animation', 'Children', 
                      'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 
                      'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 
                      'War', 'Western']
            )
            
            self.ratings_df = pd.read_csv(
                f"{temp_dir}/ml-100k/u.data",
                sep='\t',
                header=None,
                names=['user_id', 'movie_id', 'rating', 'timestamp']
            )
            
            self.users_df = pd.read_csv(
                f"{temp_dir}/ml-100k/u.user",
                sep='|',
                header=None,
                names=['user_id', 'age', 'gender', 'occupation', 'zip_code']
            )
            
        print("Dataset loaded successfully!")
            
    def load_data(self, data_path):
        #loading the MovieLens dataset from a specified path
        print(f"Loading data from {data_path}...")
        
        self.movies_df = pd.read_csv(
            f"{data_path}/u.item", 
            sep='|', 
            encoding='latin-1',
            header=None,
            names=['movie_id', 'title', 'release_date', 'video_release_date', 'IMDb_URL', 
                  'unknown', 'Action', 'Adventure', 'Animation', 'Children', 
                  'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 
                  'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 
                  'War', 'Western']
        )
        
        self.ratings_df = pd.read_csv(
            f"{data_path}/u.data",
            sep='\t',
            header=None,
            names=['user_id', 'movie_id', 'rating', 'timestamp']
        )
        
        self.users_df = pd.read_csv(
            f"{data_path}/u.user",
            sep='|',
            header=None,
            names=['user_id', 'age', 'gender', 'occupation', 'zip_code']
        )
        
        print("Dataset loaded successfully!")
    
    def preprocess_data(self):
        #preprocessing the loaded data
        print("Preprocessing data...")
        
        #cleaning the movie titles through extracting year and cleaning title
        self.movies_df['year'] = self.movies_df['title'].str.extract(r'\((\d{4})\)$', expand=False)
        self.movies_df['clean_title'] = self.movies_df['title'].str.replace(r'\s*\(\d{4}\)$', '', regex=True)
        
        #extracting the genres as a list & creating a combined feature string
        genre_columns = self.movies_df.columns[5:24]
        self.movies_df['genres'] = self.movies_df[genre_columns].apply(
            lambda x: [genre_columns[i].lower() for i, val in enumerate(x) if val == 1], 
            axis=1
        )
        
        #creating a combined feature string for the content-based filtering
        self.movies_df['combined_features'] = self.movies_df.apply(
            lambda x: f"{x['clean_title']} {' '.join(x['genres'])}", 
            axis=1
        )
        
        #computing the average rating for each movie in dataset
        movie_ratings = self.ratings_df.groupby('movie_id')['rating'].agg(['mean', 'count'])
        self.movies_df = pd.merge(self.movies_df, movie_ratings, left_on='movie_id', right_index=True, how='left')
        self.movies_df.rename(columns={'mean': 'avg_rating', 'count': 'rating_count'}, inplace=True)
        
        #filling the missing values
        self.movies_df['avg_rating'].fillna(0, inplace=True)
        self.movies_df['rating_count'].fillna(0, inplace=True)
        
    def create_movie_features(self):
        #creating feature vectors for the movies by TF-IDF
        print("Creating movie feature vectors...")
        
        #creating TF-IDF vectors for the combined features
        tfidf = TfidfVectorizer(stop_words='english')
        self.movie_features = tfidf.fit_transform(self.movies_df['combined_features'])
        
        #computing the cosine similarity between movies
        self.similarity_matrix = cosine_similarity(self.movie_features)
        
        print("Feature vectors and similarity matrix created!")
    
    def get_top_movies(self, n=100, min_ratings=10):
        #getting the top N movies based on average rating with minimum number of ratings
        popular_movies = self.movies_df[self.movies_df['rating_count'] >= min_ratings]
        return popular_movies.sort_values('avg_rating', ascending=False).head(n)
    
    def add_user_preference(self, movie_id):
        #adding a movie to user preferences 
        if movie_id not in self.user_preferences:
            self.user_preferences.append(movie_id)
            print(f"Movie '{self.movies_df[self.movies_df['movie_id'] == movie_id]['title'].values[0]}' added to preferences.")
        else:
            print("This movie is already in your preferences.")
    
    def remove_user_preference(self, movie_id):
        #removing a movie from the user preferences
        if movie_id in self.user_preferences:
            self.user_preferences.remove(movie_id)
            print(f"Movie removed from preferences.")
        else:
            print("This movie is not in your preferences.")
    
    def get_user_preferences(self):
        #get the list of the movies in the user preferences
        if not self.user_preferences:
            print("You haven't added any preferences yet.")
            return None
        
        preferences_df = self.movies_df[self.movies_df['movie_id'].isin(self.user_preferences)]
        return preferences_df[['movie_id', 'title', 'avg_rating', 'rating_count']]
    
    def content_based_recommendations(self, n=10):
        #generating the content-based recommendations based on user preferences
        if not self.user_preferences:
            print("You need to add some movie preferences first.")
            return None
        
        # geting the indices of user's preferred movies
        preference_indices = [self.movies_df[self.movies_df['movie_id'] == movie_id].index[0] 
                              for movie_id in self.user_preferences]
        
        # calculating the similarity scores for all movies compared to user preferences
        similar_movies = {}
        for idx in preference_indices:
            similar_movies_dict = {i: score for i, score in enumerate(self.similarity_matrix[idx])}
            for movie_idx, score in similar_movies_dict.items():
                if movie_idx in similar_movies:
                    similar_movies[movie_idx] += score
                else:
                    similar_movies[movie_idx] = score
        
        # sorting & geting  top recommendations without the already preferred movies selected by the user
        similar_movies = {k: v / len(preference_indices) for k, v in similar_movies.items()}
        similar_movies = sorted(similar_movies.items(), key=lambda x: x[1], reverse=True)
        
        # filters the movies that are already in preferences
        recommendations = []
        for movie_idx, score in similar_movies:
            if self.movies_df.iloc[movie_idx]['movie_id'] not in self.user_preferences:
                recommendations.append((movie_idx, score))
                if len(recommendations) >= n:
                    break
        
        # geting the details of recommended movies
        recommended_movies = pd.DataFrame([
            {
                'movie_id': self.movies_df.iloc[idx]['movie_id'],
                'title': self.movies_df.iloc[idx]['title'],
                'genres': ', '.join(self.movies_df.iloc[idx]['genres']),
                'avg_rating': self.movies_df.iloc[idx]['avg_rating'],
                'similarity_score': score
            }
            for idx, score in recommendations
        ])
        
        return recommended_movies
    
    def collaborative_recommendations(self, n=10):
        #generating a collaborative filtering recommendations from the user preferences
        if not self.user_preferences:
            print("You need to add some movie preferences first.")
            return None
        
        # creating a user-item matrix
        user_item_matrix = self.ratings_df.pivot_table(
            index='user_id', 
            columns='movie_id', 
            values='rating'
        ).fillna(0)
        
        # for each of the preferred movie, find users who rated it highly
        similar_users = set()
        for movie_id in self.user_preferences:
            # finding the users who rated this movie 4 or higher
            high_raters = self.ratings_df[(self.ratings_df['movie_id'] == movie_id) & 
                                     (self.ratings_df['rating'] >= 4)]['user_id'].tolist()
            similar_users.update(high_raters)
        
        if not similar_users:
            print("No similar users found. Try adding different movies to your preferences.")
            return None
        
        # finding the movies that similar users rated highly, but still not in preferences
        recommendations = {}
        for user_id in similar_users:
            # getting high rated movies from this user
            high_rated = self.ratings_df[(self.ratings_df['user_id'] == user_id) & 
                                    (self.ratings_df['rating'] >= 4)]['movie_id'].tolist()
            
            # adding to recommendations if not in user preferences
            for movie_id in high_rated:
                if movie_id not in self.user_preferences:
                    if movie_id in recommendations:
                        recommendations[movie_id] += 1
                    else:
                        recommendations[movie_id] = 1
        
        # sorting recommendations by frequency
        recommendations = sorted(recommendations.items(), key=lambda x: x[1], reverse=True)[:n]
        
        # getting the details of recommended movies
        recommended_movies = pd.DataFrame([
            {
                'movie_id': movie_id,
                'title': self.movies_df[self.movies_df['movie_id'] == movie_id]['title'].values[0],
                'genres': ', '.join(self.movies_df[self.movies_df['movie_id'] == movie_id]['genres'].values[0]),
                'avg_rating': self.movies_df[self.movies_df['movie_id'] == movie_id]['avg_rating'].values[0],
                'recommendation_score': score
            }
            for movie_id, score in recommendations
        ])
        
        return recommended_movies
    
    def hybrid_recommendations(self, n=10, content_weight=0.6):
        #generating the  recommendations by combining content-based & collaborative approaches
        content_recs = self.content_based_recommendations(n=n*2)
        collab_recs = self.collaborative_recommendations(n=n*2)
        
        if content_recs is None or collab_recs is None:
            return content_recs if collab_recs is None else collab_recs
        
        # normalizing the similarity scores
        content_recs['norm_score'] = content_recs['similarity_score'] / content_recs['similarity_score'].max()
        collab_recs['norm_score'] = collab_recs['recommendation_score'] / collab_recs['recommendation_score'].max()
        
        # combining recommendations
        all_recs = pd.concat([
            content_recs[['movie_id', 'title', 'genres', 'avg_rating', 'norm_score']].assign(source='content'),
            collab_recs[['movie_id', 'title', 'genres', 'avg_rating', 'norm_score']].assign(source='collab')
        ])
        
        # applying weights based on source
        all_recs['weighted_score'] = all_recs.apply(
            lambda x: x['norm_score'] * content_weight if x['source'] == 'content' 
            else x['norm_score'] * (1 - content_weight),
            axis=1
        )
        
        # grouping by movie & sum the weighted scores
        hybrid_recs = all_recs.groupby(['movie_id', 'title', 'genres', 'avg_rating'])[['weighted_score']].sum()
        hybrid_recs = hybrid_recs.reset_index().sort_values('weighted_score', ascending=False).head(n)
        
        return hybrid_recs
    
    def visualize_recommendations(self, recommendations):
        if recommendations is None or len(recommendations) == 0:
            print("No recommendations to visualize.")
            return
    
        plt.figure(figsize=(12, 6))
    
        # bar chart for recommendation scores
        plt.subplot(1, 2, 1)
        # determining on which score column to use based on what's available in the dataframe
        if 'similarity_score' in recommendations.columns:
            score_col = 'similarity_score'
        elif 'recommendation_score' in recommendations.columns:
            score_col = 'recommendation_score'
        elif 'weighted_score' in recommendations.columns:
            score_col = 'weighted_score'
        else:
            print("No score column found in recommendations.")
            return
        
        recommendations = recommendations.sort_values(score_col, ascending=True)
        plt.barh(recommendations['title'].str.slice(0, 30), recommendations[score_col])
        plt.xlabel('Recommendation Score')
        plt.title('Movie Recommendations')
        plt.tight_layout()
    
        # the bubble chart of ratings-recommendation score
        plt.subplot(1, 2, 2)
        plt.scatter(
            recommendations['avg_rating'], 
            recommendations[score_col],
            s=recommendations['avg_rating'] * 30,  
            alpha=0.6
        )
        plt.xlabel('Average Rating')
        plt.ylabel('Recommendation Score')
        plt.title('Rating vs. Recommendation Score')
    
        for i, row in recommendations.iterrows():
            plt.annotate(
                row['title'][:20] + '...' if len(row['title']) > 20 else row['title'],
                (row['avg_rating'], row[score_col]),
                fontsize=8
            )
    
        plt.tight_layout()
        plt.show()


# the user interface class
class MovieRecommendationUI:
    
    def __init__(self):
        self.recommender = MovieRecommendationSystem()
        
    def display_menu(self):
        #displaying the main menu
        print("\n" + "="*50)
        print("MOVIE RECOMMENDATION SYSTEM")
        print("="*50)
        print("1. Browse Popular Movies")
        print("2. Search for a Movie")
        print("3. View Your Preferences")
        print("4. Get Content-Based Recommendations")
        print("5. Get Collaborative Recommendations")
        print("6. Get Hybrid Recommendations")
        print("7. Exit")
        print("="*50)
        
    def run(self):
        #running the main application loop
        print("\nWelcome to the Movie Recommendation System!")
        print("Loading data and preparing the system...")
        
        while True:
            self.display_menu()
            choice = input("\nEnter your choice (1-7): ")
            
            if choice == '1':
                self.browse_popular_movies()
            elif choice == '2':
                self.search_movies()
            elif choice == '3':
                self.view_preferences()
            elif choice == '4':
                self.get_content_recommendations()
            elif choice == '5':
                self.get_collaborative_recommendations()
            elif choice == '6':
                self.get_hybrid_recommendations()
            elif choice == '7':
                print("\nThank you for using the Movie Recommendation System. Goodbye!")
                break
            else:
                print("\nInvalid choice. Please try again.")
    
    def browse_popular_movies(self):
        #browsing the popular movies & adding them to preferences by user choice
        top_n = input("\nHow many top movies would you like to see? (default: 20): ")
        top_n = int(top_n) if top_n.isdigit() else 20
        
        min_ratings = input("Minimum number of ratings? (default: 50): ")
        min_ratings = int(min_ratings) if min_ratings.isdigit() else 50
        
        popular_movies = self.recommender.get_top_movies(n=top_n, min_ratings=min_ratings)
        
        print(f"\nTop {top_n} Movies:")
        print("-" * 80)
        for i, (_, movie) in enumerate(popular_movies.iterrows(), 1):
            print(f"{i}. {movie['title']} - Rating: {movie['avg_rating']:.2f} ({movie['rating_count']} ratings)")
        
        self.add_to_preferences(popular_movies)
    
    def search_movies(self):
       #searching for movies 
        search_term = input("\nEnter a movie title to search for: ")
        
        if not search_term:
            print("Search term cannot be empty.")
            return
        
        results = self.recommender.movies_df[
            self.recommender.movies_df['title'].str.contains(search_term, case=False)
        ]
        
        if results.empty:
            print("No movies found matching that title.")
            return
        
        print(f"\nFound {len(results)} movies:")
        print("-" * 80)
        for i, (_, movie) in enumerate(results.iterrows(), 1):
            print(f"{i}. {movie['title']} - Rating: {movie['avg_rating']:.2f} ({movie['rating_count']} ratings)")
        
        self.add_to_preferences(results)
    
    def add_to_preferences(self, movie_list):
        #add selected movies that the user chose to preferences
        while True:
            choice = input("\nEnter the number of a movie to add to your preferences (or 'q' to go back): ")
            
            if choice.lower() == 'q':
                break
            
            if choice.isdigit() and 1 <= int(choice) <= len(movie_list):
                idx = int(choice) - 1
                movie_id = movie_list.iloc[idx]['movie_id']
                self.recommender.add_user_preference(movie_id)
            else:
                print("Invalid choice. Please try again.")
    
    def view_preferences(self):
        #view the user preferences
        preferences = self.recommender.get_user_preferences()
        
        if preferences is not None and not preferences.empty:
            print("\nYour Movie Preferences:")
            print("-" * 80)
            for i, (_, movie) in enumerate(preferences.iterrows(), 1):
                print(f"{i}. {movie['title']} - Rating: {movie['avg_rating']:.2f}")
            
            # the option for the users remove a movie from preferences
            choice = input("\nEnter the number of a movie to remove from preferences (or 'q' to go back): ")
            if choice.lower() != 'q' and choice.isdigit() and 1 <= int(choice) <= len(preferences):
                idx = int(choice) - 1
                movie_id = preferences.iloc[idx]['movie_id']
                self.recommender.remove_user_preference(movie_id)
    
    def get_content_recommendations(self):
        # gather & display the content-based recommendations 
        if not self.recommender.user_preferences:
            print("\nPlease add some movies to your preferences first.")
            return
        
        n = input("\nHow many recommendations would you like? (default: 10): ")
        n = int(n) if n.isdigit() else 10
        
        print("\nGenerating content-based recommendations...")
        recommendations = self.recommender.content_based_recommendations(n=n)
        
        if recommendations is not None and not recommendations.empty:
            print("\nHere are your recommendations:")
            print("-" * 80)
            for i, (_, movie) in enumerate(recommendations.iterrows(), 1):
                print(f"{i}. {movie['title']} - Rating: {movie['avg_rating']:.2f} - Similarity: {movie['similarity_score']:.4f}")
            
            # ask the user if he/she wants to visualize
            if input("\nWould you like to visualize these recommendations? (y/n): ").lower() == 'y':
                self.recommender.visualize_recommendations(recommendations)
    
    def get_collaborative_recommendations(self):
        #getting & displaying the collaborative filtering recommendations
        if not self.recommender.user_preferences:
            print("\nPlease add some movies to your preferences first.")
            return
        
        n = input("\nHow many recommendations would you like? (default: 10): ")
        n = int(n) if n.isdigit() else 10
        
        print("\nGenerating collaborative filtering recommendations...")
        recommendations = self.recommender.collaborative_recommendations(n=n)
        
        if recommendations is not None and not recommendations.empty:
            print("\nHere are your recommendations:")
            print("-" * 80)
            for i, (_, movie) in enumerate(recommendations.iterrows(), 1):
                print(f"{i}. {movie['title']} - Rating: {movie['avg_rating']:.2f} - Score: {movie['recommendation_score']}")
            
            # ask the user if he/she wants to visualize
            if input("\nWould you like to visualize these recommendations? (y/n): ").lower() == 'y':
                self.recommender.visualize_recommendations(recommendations)
    
    def get_hybrid_recommendations(self):
        # getting & displaying the hybrid recommendations
        if not self.recommender.user_preferences:
            print("\nPlease add some movies to your preferences first.")
            return
        
        n = input("\nHow many recommendations would you like? (default: 10): ")
        n = int(n) if n.isdigit() else 10
        
        weight = input("\nContent-based weight (0.0 to 1.0, default: 0.6): ")
        try:
            weight = float(weight) if weight else 0.6
            if not 0 <= weight <= 1:
                weight = 0.6
        except ValueError:
            weight = 0.6
        
        print("\nGenerating hybrid recommendations...")
        recommendations = self.recommender.hybrid_recommendations(n=n, content_weight=weight)
        
        if recommendations is not None and not recommendations.empty:
            print("\nHere are your recommendations:")
            print("-" * 80)
            for i, (_, movie) in enumerate(recommendations.iterrows(), 1):
                print(f"{i}. {movie['title']} - Rating: {movie['avg_rating']:.2f} - Score: {movie['weighted_score']:.4f}")
            
            # ask the user if uhe/she wants to visualize
            if input("\nWould you like to visualize these recommendations? (y/n): ").lower() == 'y':
                self.recommender.visualize_recommendations(recommendations)


# the main entry point
if __name__ == "__main__":
    # initializing & running  the UI
    ui = MovieRecommendationUI()
    ui.run()