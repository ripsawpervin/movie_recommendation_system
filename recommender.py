"""Content-based recommendations: TF-IDF followed by cosine similarity."""
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class MovieRecommender:
    def __init__(self, data):
        self.data = data
        df = data.movies
        features = (df['genre_text'] + ' ' + df['overview'] + ' ' +
                    df['keyword_list'].apply(' '.join))
        self.vectorizer = TfidfVectorizer(stop_words='english', max_features=20000,
                                          ngram_range=(1, 2), sublinear_tf=True)
        try:
            self.matrix = self.vectorizer.fit_transform(features)
        except ValueError:
            # A malformed replacement dataset may have no usable content.
            self.matrix = self.vectorizer.fit_transform(['unknowncontent'] * len(df))
        self.positions = {int(movie_id): index for index, movie_id in enumerate(df['id'])}

    def _rank(self, candidates, scores, limit=5):
        result = candidates.copy()
        positions = [self.positions[int(movie_id)] for movie_id in result['id']]
        result['similarity'] = np.clip(np.asarray(scores)[positions], 0, 1)
        return result.sort_values(['similarity', 'vote_average', 'sort_title', 'id'],
                                 ascending=[False, False, True, True]).head(limit)

    def similar(self, movie_id, language='All', min_rating=0):
        if int(movie_id) not in self.positions:
            raise ValueError('Select a valid movie first.')
        candidates = self.data.filter_movies(language=language, min_rating=min_rating)
        candidates = candidates[candidates['id'] != int(movie_id)]
        index = self.positions[int(movie_id)]
        # Compute one row on demand, rather than a large all-pairs matrix.
        scores = cosine_similarity(self.matrix[index], self.matrix).ravel()
        return self._rank(candidates, scores)

    def preferences(self, genres=(), language='All', min_rating=0, keywords=''):
        query = ' '.join(genres) + ' ' + keywords.strip()
        if not query.strip():
            raise ValueError('Choose a genre or enter movie-related keywords.')
        candidates = self.data.filter_movies(genres=genres, language=language, min_rating=min_rating)
        vector = self.vectorizer.transform([query])
        scores = cosine_similarity(vector, self.matrix).ravel()
        result = self._rank(candidates, scores, limit=len(candidates))
        return result[result['similarity'] > 0].head(5)
