"""Load the CSV and manage local favorites/history. No network is used."""
import json
from pathlib import Path
import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent / 'data'


def extract_names(value):
    """TMDB stores genres and keywords as JSON lists of dictionaries."""
    try:
        items = json.loads(value) if isinstance(value, str) else []
        if not isinstance(items, list):
            return []
        return [item['name'].strip() for item in items
                if isinstance(item, dict) and isinstance(item.get('name'), str)
                and item['name'].strip()]
    except (ValueError, TypeError):
        return []


class DataManager:
    def __init__(self, data_dir=DATA_DIR):
        self.data_dir = Path(data_dir)
        self.warnings = []
        path = self.data_dir / 'movies.csv'
        if not path.exists():
            raise ValueError(f'Movie dataset missing. Put movies.csv inside:\n{self.data_dir}')
        try:
            df = pd.read_csv(path)
        except (OSError, ValueError) as error:
            raise ValueError(f'Cannot read movies.csv: {error}') from error
        required = {'id', 'title', 'genres', 'overview', 'vote_average'}
        if not required.issubset(df.columns):
            raise ValueError('CSV is missing columns: ' + ', '.join(sorted(required - set(df.columns))))
        for column in ['keywords', 'original_language', 'release_date']:
            if column not in df:
                df[column] = ''
        for column in ['title', 'overview', 'original_language']:
            df[column] = df[column].fillna('').astype(str).str.strip()
        ids = pd.to_numeric(df['id'], errors='coerce')
        valid = ids.notna() & np.isfinite(ids) & (ids >= 0) & (ids % 1 == 0) & (ids < 2**63)
        df = df.loc[valid & df['title'].ne('')].copy()
        df['id'] = ids.loc[df.index].astype('int64')
        df = df.drop_duplicates('id').reset_index(drop=True)
        if df.empty:
            raise ValueError('The dataset has no valid movies.')
        for column in ['vote_average', 'runtime', 'popularity']:
            if column not in df:
                df[column] = 0
            df[column] = pd.to_numeric(df[column], errors='coerce').replace([np.inf, -np.inf], np.nan).fillna(0).clip(lower=0)
        df['vote_average'] = df['vote_average'].clip(upper=10)
        df['year'] = pd.to_datetime(df['release_date'], errors='coerce').dt.year.fillna(0).astype(int)
        df['genre_list'] = df['genres'].apply(extract_names)
        df['keyword_list'] = df['keywords'].apply(extract_names)
        df['genre_text'] = df['genre_list'].apply(', '.join)
        df['original_language'] = df['original_language'].replace('', 'unknown')
        df['sort_title'] = df['title'].str.casefold()
        self.movies = df
        self.genres = sorted({genre for names in df['genre_list'] for genre in names})
        self.languages = sorted(df['original_language'].unique())
        self.favorites = self._load_ids('favorites.json')
        self.history = self._load_ids('history.json')[:10]

    def _load_ids(self, name):
        path = self.data_dir / name
        if not path.exists():
            return []
        try:
            values = json.loads(path.read_text(encoding='utf-8'))
            if not isinstance(values, list):
                raise ValueError('Expected a JSON list.')
            known = set(self.movies['id'])
            result = []
            for value in values:
                if type(value) is int and value in known and value not in result:
                    result.append(value)
            return result
        except (OSError, ValueError) as error:
            self.warnings.append(f'{name} could not be loaded; starting with an empty list. {error}')
            return []

    def _save_ids(self, name, values):
        # Replace the file only after the complete new JSON has been written.
        self.data_dir.mkdir(parents=True, exist_ok=True)
        path = self.data_dir / name
        temporary = path.with_suffix('.tmp')
        temporary.write_text(json.dumps(values, indent=2), encoding='utf-8')
        temporary.replace(path)

    def movie(self, movie_id):
        rows = self.movies[self.movies['id'] == int(movie_id)]
        if rows.empty:
            raise ValueError('Movie not found.')
        return rows.iloc[0]

    def set_favorite(self, movie_id, add=True):
        movie_id = int(self.movie(movie_id)['id'])
        values = [value for value in self.favorites if value != movie_id]
        if add:
            values.append(movie_id)
        self._save_ids('favorites.json', values)
        self.favorites = values

    def view(self, movie_id):
        movie_id = int(self.movie(movie_id)['id'])
        values = ([movie_id] + [value for value in self.history if value != movie_id])[:10]
        self._save_ids('history.json', values)
        self.history = values

    def by_ids(self, ids):
        return self.movies.set_index('id', drop=False).reindex(ids).dropna(subset=['title']).reset_index(drop=True)

    def filter_movies(self, search='', genres=(), min_rating=0, language='All',
                      year_from=None, year_to=None, sort='Rating'):
        df = self.movies
        if search.strip():
            df = df[df['title'].str.contains(search.strip(), case=False, regex=False)]
        if genres:
            selected = set(genres)
            # Multiple selections mean ANY of the selected genres.
            df = df[df['genre_list'].apply(lambda names: bool(selected.intersection(names)))]
        df = df[df['vote_average'] >= min_rating]
        if language != 'All':
            df = df[df['original_language'] == language]
        if year_from is not None:
            df = df[df['year'] >= year_from]
        if year_to is not None:
            df = df[(df['year'] > 0) & (df['year'] <= year_to)]
        column = {'Rating': 'vote_average', 'Year': 'year', 'Title': 'sort_title', 'Popularity': 'popularity'}[sort]
        return df.sort_values([column, 'id'] if column == 'sort_title' else [column, 'sort_title', 'id'],
                              ascending=[True, True] if column == 'sort_title' else [False, True, True])
