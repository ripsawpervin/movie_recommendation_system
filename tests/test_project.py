"""Run: python -m unittest discover -s tests -v"""
import json
from pathlib import Path
import shutil
import tempfile
import unittest
import numpy as np
from data_manager import DataManager, extract_names
from recommender import MovieRecommender
from analytics import create_figure


class ProjectTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = DataManager()
        cls.engine = MovieRecommender(cls.data)

    def test_dataset(self):
        self.assertEqual(len(self.data.movies), 4803)
        self.assertEqual(len(self.data.genres), 20)
        self.assertFalse(self.data.movies['overview'].isna().any())

    def test_json_columns(self):
        self.assertEqual(extract_names('[{"id": 1, "name": "Action"}]'), ['Action'])
        for invalid in ['broken', '{}', 'null', '[1, null, {}]', None]:
            self.assertEqual(extract_names(invalid), [])

    def test_literal_case_insensitive_search(self):
        self.assertEqual(self.data.filter_movies(search='AVATAR').iloc[0]['title'], 'Avatar')
        self.assertTrue(self.data.filter_movies(search='[not a regex').empty)

    def test_combined_filters(self):
        result = self.data.filter_movies(genres=['Action', 'Adventure'], language='en',
                                         min_rating=7, year_from=2000, year_to=2015)
        self.assertGreater(len(result), 0)
        self.assertTrue((result.vote_average >= 7).all())
        self.assertTrue((result.original_language == 'en').all())
        self.assertTrue(result.year.between(2000, 2015).all())
        self.assertTrue(result.genre_list.apply(lambda g: bool({'Action','Adventure'} & set(g))).all())

    def test_similarity_and_determinism(self):
        result = self.engine.similar(19995)
        self.assertEqual(len(result), 5)
        self.assertNotIn(19995, result.id.tolist())
        self.assertTrue(result.similarity.between(0, 1).all())
        self.assertTrue(result.similarity.is_monotonic_decreasing)
        self.assertEqual(result.id.tolist(), self.engine.similar(19995).id.tolist())
        self.assertEqual(result.iloc[0].title, 'Aliens')
        filtered = self.engine.similar(19995, language='en', min_rating=8)
        self.assertTrue((filtered.vote_average >= 8).all())
        self.assertTrue((filtered.original_language == 'en').all())

    def test_preferences_and_empty_matches(self):
        result = self.engine.preferences(['Science Fiction'], 'en', 7, 'space adventure')
        self.assertEqual(len(result), 5)
        self.assertTrue(result.genre_list.apply(lambda g: 'Science Fiction' in g).all())
        self.assertTrue((result.similarity > 0).all())
        self.assertTrue(self.engine.preferences(keywords='zzzzzzzzzzzzzzzz').empty)
        self.assertTrue(self.engine.preferences(['Action'], language='nonexistent').empty)
        with self.assertRaises(ValueError):
            self.engine.preferences()

    def test_tie_breaking(self):
        result = self.engine._rank(self.data.movies, np.zeros(len(self.data.movies)))
        expected = self.data.movies.sort_values(['vote_average','sort_title','id'], ascending=[False, True, True]).head(5)
        self.assertEqual(result.id.tolist(), expected.id.tolist())

    def test_persistence_and_history_limit(self):
        with tempfile.TemporaryDirectory() as folder:
            shutil.copy(self.data.data_dir / 'movies.csv', Path(folder) / 'movies.csv')
            data = DataManager(folder)
            data.set_favorite(19995)
            data.set_favorite(19995)
            self.assertEqual(data.favorites, [19995])
            ids = list(map(int, data.movies.head(12).id))
            for movie_id in ids:
                data.view(movie_id)
            self.assertEqual(data.history, ids[::-1][:10])
            loaded = DataManager(folder)
            self.assertEqual(loaded.favorites, [19995])
            self.assertEqual(loaded.history, data.history)
            loaded.set_favorite(19995, False)
            self.assertEqual(json.loads((Path(folder)/'favorites.json').read_text()), [])
            (Path(folder)/'history.json').write_text('broken')
            self.assertEqual(DataManager(folder).history, [])

    def test_invalid_dataset(self):
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaises(ValueError):
                DataManager(folder)
            (Path(folder)/'movies.csv').write_text('title\nTest\n')
            with self.assertRaises(ValueError):
                DataManager(folder)

    def test_figures(self):
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        for kind in ['Genres', 'Ratings', 'Top 10']:
            figure = create_figure(self.data.movies, kind)
            FigureCanvasAgg(figure).draw()
            self.assertEqual(len(figure.axes), 1)
            figure.clear()


if __name__ == '__main__':
    unittest.main()
