# AI-Based Movie Recommendation System Using Python

## 1. Introduction and objective
CineMatch is an offline desktop application for an MCA Python mini-project. It helps users browse movies and find content-based recommendations using genres, descriptions and keywords. It uses the supplied TMDB CSV containing **4,803 movies and 20 original columns**. The interface uses a dark navy theme, a sidebar and readable movie tables.

The goal is to demonstrate understandable Python programming, Pandas data handling and a real text-similarity algorithm. No API key, Ollama, large language model, web server or account is required. Installing dependencies normally needs internet once; running the application does not.

## 2. Functional system architecture

| Layer | File | Responsibility |
|---|---|---|
| Startup | `main.py` | Create Tkinter window, load data/model, handle startup errors |
| Interface | `gui.py` | Sidebar, forms, tables, details, favorites, history and chart navigation |
| Data | `data_manager.py` | Read/clean CSV, parse JSON columns, filter records, save JSON lists |
| Recommendation engine | `recommender.py` | Build TF-IDF vectors and rank cosine similarities |
| Visualization | `analytics.py` | Build three Matplotlib figures for embedding in Tkinter |
| Storage | `data/` | Original movie CSV and local favorites/history |

Data flow: startup loads CSV → data manager cleans records → recommender builds a sparse TF-IDF matrix → GUI sends user choices → engine applies filters and scores candidates → GUI displays up to five results. Selecting a movie sends its ID to the history store. Favorite actions update the favorites store. Analytics reads the cleaned DataFrame.

The model is fitted once per launch. Each similar-movie request computes one similarity row, avoiding an unnecessary 4,803 × 4,803 matrix. History and favorites are storage features; they do not train or personalize the model automatically.

## 3. Complete folder structure
```text
movie_recommendation_system/
├── main.py
├── gui.py
├── recommender.py
├── data_manager.py
├── analytics.py
├── data/
│   ├── movies.csv
│   ├── favorites.json
│   └── history.json
├── requirements.txt
├── README.md
├── COMPLETE_PROJECT_GUIDE.md
└── tests/
    └── test_project.py
```

## 4. Movie dataset
The complete attached CSV is included unchanged as `data/movies.csv`; it is not a small demonstration dataset. It has 4,803 rows, 20 columns, three missing overviews and no missing titles. Cleaning occurs in memory. There are 20 distinct genres and 85 movies with dataset ratings of at least 8.

| Column | How it is used |
|---|---|
| `id` | Stable movie identifier; favorite/history storage and table row ID |
| `title` | Search, display and deterministic sorting |
| `genres` | JSON name extraction, genre filters and text features |
| `overview` | Description and text features; missing values become empty text |
| `keywords` | JSON name extraction and additional text features |
| `vote_average` | Minimum rating filter, display and ranking tie-breaker |
| `release_date` | Extract release year; invalid/missing dates become unknown |
| `original_language` | Language filter using dataset codes |
| `runtime` | Display duration in minutes |
| `popularity` | Optional browse sort |

Example: `[{"id": 28, "name": "Action"}]` becomes the list `['Action']`. Parsing uses `json.loads`, not `eval`. Invalid JSON becomes an empty list. Invalid movie IDs or empty titles are skipped; duplicate IDs keep the first record. Numeric values are converted safely. An unknown year is excluded when a year filter is applied.

CSV provenance: the user-supplied TMDB 5000 movie dataset. This package does not establish redistribution rights; consult the original dataset terms before public redistribution.

## 5. requirements.txt
```text
pandas>=2.2,<3
numpy>=1.26,<3
scikit-learn>=1.4,<2
matplotlib>=3.8,<4
```
Tkinter, pathlib, json, math and unittest belong to Python's standard library. Tkinter must be included in the Python installation; do not install a package called tkinter using pip.

## 6. Full source code

### main.py
```python
"""Run with: python main.py"""
import tkinter as tk
from tkinter import messagebox


def main():
    root = tk.Tk()
    root.withdraw()
    try:
        from data_manager import DataManager
        from recommender import MovieRecommender
        from gui import MovieApp
        data = DataManager()
        engine = MovieRecommender(data)
        MovieApp(root, data, engine)
    except Exception as error:
        messagebox.showerror('Startup problem', f'{error}\n\nCheck the dataset and install requirements.txt.', parent=root)
        root.destroy()
        return
    root.deiconify()
    if data.warnings:
        messagebox.showwarning('Saved lists', '\n'.join(data.warnings), parent=root)
    root.mainloop()


if __name__ == '__main__':
    main()
```

### gui.py
```python
"""Simple Tkinter pages sharing one movie table and one details panel."""
import math
import tkinter as tk
from tkinter import ttk, messagebox
from tkinter.scrolledtext import ScrolledText
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from analytics import create_figure

BG = '#0b1220'
PANEL = '#111c30'
TEXT = '#e5edff'
MUTED = '#a6b8d5'
BLUE = '#2478df'


class MovieApp:
    def __init__(self, root, data, engine):
        self.root, self.data, self.engine = root, data, engine
        self.selected_id = None
        self.chart = None
        root.title('CineMatch | AI-Based Movie Recommendation System')
        root.geometry('1180x760')
        root.minsize(960, 650)
        root.configure(bg=BG)
        style = ttk.Style(root)
        style.theme_use('clam')
        style.configure('.', font=('Segoe UI', 10), background=BG, foreground=TEXT)
        style.configure('TFrame', background=BG)
        style.configure('TLabel', background=BG, foreground=TEXT)
        style.configure('Title.TLabel', font=('Segoe UI', 23, 'bold'))
        style.configure('Muted.TLabel', foreground=MUTED)
        style.configure('TButton', background=BLUE, foreground='white', padding=(10, 7), borderwidth=0)
        style.map('TButton', background=[('active', '#398df4'), ('disabled', '#344563')])
        style.configure('TEntry', fieldbackground=PANEL, foreground=TEXT, insertcolor=TEXT)
        style.configure('TCombobox', fieldbackground=PANEL, foreground=TEXT, arrowcolor=TEXT)
        style.map('TCombobox', fieldbackground=[('readonly', PANEL)], foreground=[('readonly', TEXT)],
                  selectbackground=[('readonly', PANEL)], selectforeground=[('readonly', TEXT)])
        style.configure('Treeview', background=PANEL, fieldbackground=PANEL, foreground=TEXT,
                        rowheight=30, borderwidth=0)
        style.configure('Treeview.Heading', background='#20324f', foreground=TEXT, padding=8)
        style.map('Treeview', background=[('selected', BLUE)], foreground=[('selected', 'white')])
        root.option_add('*TCombobox*Listbox.background', PANEL)
        root.option_add('*TCombobox*Listbox.foreground', TEXT)
        sidebar = tk.Frame(root, bg=PANEL, width=190)
        sidebar.pack(side='left', fill='y')
        sidebar.pack_propagate(False)
        tk.Label(sidebar, text='CINEMATCH', bg=PANEL, fg='#67acff', font=('Segoe UI', 19, 'bold')).pack(pady=(28, 3))
        tk.Label(sidebar, text='Your offline movie guide', bg=PANEL, fg=MUTED).pack(pady=(0, 25))
        for label, command in [('Dashboard', self.dashboard), ('Browse Movies', self.browse),
                               ('AI Recommendations', self.recommendations), ('Favorites', self.favorites),
                               ('History', self.history), ('Analytics', self.analytics), ('About & Help', self.about)]:
            ttk.Button(sidebar, text=label, command=command).pack(fill='x', padx=12, pady=5)
        tk.Label(sidebar, text='TF-IDF + cosine similarity\nLocal data • No API', bg=PANEL, fg=MUTED,
                 justify='left').pack(side='bottom', pady=25)
        self.status = tk.StringVar(value='Ready')
        body = ttk.Frame(root, padding=20)
        body.pack(side='left', fill='both', expand=True)
        ttk.Label(body, textvariable=self.status, style='Muted.TLabel', wraplength=740).pack(side='bottom', fill='x', pady=(10, 0))
        self.page = ttk.Frame(body)
        self.page.pack(fill='both', expand=True)
        self.dashboard()

    def clear(self, title, subtitle):
        if self.chart is not None:
            self.chart.figure.clear()
            self.chart = None
        for widget in self.page.winfo_children():
            widget.destroy()
        self.selected_id = None
        self.tree = None
        self.status.set('Ready')
        ttk.Label(self.page, text=title, style='Title.TLabel').pack(anchor='w')
        ttk.Label(self.page, text=subtitle, style='Muted.TLabel', wraplength=740).pack(anchor='w', pady=(4, 14))

    def dashboard(self):
        self.clear('Discover your next movie', 'Welcome to CineMatch. Explore your collection and find movies with similar content.')
        cards = ttk.Frame(self.page)
        cards.pack(fill='x', pady=15)
        values = [('AVAILABLE MOVIES', f'{len(self.data.movies):,}'), ('GENRES', len(self.data.genres)),
                  ('RATED 8 OR ABOVE', int((self.data.movies['vote_average'] >= 8).sum()))]
        for column, (label, value) in enumerate(values):
            cards.columnconfigure(column, weight=1)
            card = tk.Frame(cards, bg=PANEL, padx=16, pady=22)
            card.grid(row=0, column=column, sticky='nsew', padx=(0, 8))
            tk.Label(card, text=label, bg=PANEL, fg=MUTED, font=('Segoe UI', 9)).pack(anchor='w')
            tk.Label(card, text=str(value), bg=PANEL, fg='#67acff', font=('Segoe UI', 30, 'bold')).pack(anchor='w')
        ttk.Button(self.page, text='Browse the movie collection', command=self.browse).pack(anchor='w', pady=12)
        ttk.Button(self.page, text='Get personalized recommendations', command=self.recommendations).pack(anchor='w')
        ttk.Label(self.page, text='How it works', font=('Segoe UI', 16, 'bold')).pack(anchor='w', pady=(35, 10))
        ttk.Label(self.page, text='1. Search for a movie, or choose your favorite genres.\n\n'
                  '2. Set your language and minimum rating.\n\n'
                  '3. Explore five content-based matches and save your favorites.',
                  wraplength=700).pack(anchor='w')
        ttk.Label(self.page, text='Similarity measures shared content. It does not predict whether you will like a movie.\n'
                  'Favorites and history are saved locally; they do not train the recommendation model.',
                  style='Muted.TLabel', wraplength=700).pack(anchor='w', pady=30)

    def filters(self, parent, browsing=False):
        form = ttk.Frame(parent)
        form.pack(fill='x', pady=(0, 10))
        self.genre_box = tk.Listbox(form, selectmode='multiple', exportselection=False, height=4,
                                   bg=PANEL, fg=TEXT, selectbackground=BLUE, relief='flat', width=23)
        ttk.Label(form, text='Genres (click to select; ANY match)').grid(row=0, column=0, sticky='w', padx=(0, 12))
        self.genre_box.grid(row=1, column=0, rowspan=3, sticky='nsw', padx=(0, 12))
        for genre in self.data.genres:
            self.genre_box.insert('end', genre)
        scroll = ttk.Scrollbar(form, orient='vertical', command=self.genre_box.yview)
        scroll.grid(row=1, column=1, rowspan=3, sticky='ns', padx=(0, 15))
        self.genre_box.configure(yscrollcommand=scroll.set)
        self.language = tk.StringVar(value='All')
        self.rating = tk.StringVar(value='0')
        ttk.Label(form, text='Language code').grid(row=0, column=2, sticky='w')
        ttk.Combobox(form, textvariable=self.language, values=['All'] + self.data.languages,
                     state='readonly', width=14).grid(row=1, column=2, sticky='w', padx=(0, 12))
        ttk.Label(form, text='Minimum rating (0–10)').grid(row=0, column=3, sticky='w')
        ttk.Entry(form, textvariable=self.rating, width=17).grid(row=1, column=3, sticky='w')
        if browsing:
            self.year_from, self.year_to = tk.StringVar(), tk.StringVar()
            ttk.Label(form, text='From year (optional)').grid(row=2, column=2, sticky='w', pady=(5, 0))
            ttk.Label(form, text='To year (optional)').grid(row=2, column=3, sticky='w', pady=(5, 0))
            ttk.Entry(form, textvariable=self.year_from, width=17).grid(row=3, column=2, sticky='w')
            ttk.Entry(form, textvariable=self.year_to, width=17).grid(row=3, column=3, sticky='w')
        else:
            self.keywords = tk.StringVar()
            ttk.Label(form, text='Optional keywords (e.g. space adventure)').grid(row=2, column=2, columnspan=2, sticky='w', pady=(5, 0))
            ttk.Entry(form, textvariable=self.keywords, width=38).grid(row=3, column=2, columnspan=2, sticky='ew')

    def read_filters(self, browsing=False):
        try:
            rating = float(self.rating.get())
            if not math.isfinite(rating) or not 0 <= rating <= 10:
                raise ValueError
        except ValueError:
            raise ValueError('Minimum rating must be a number between 0 and 10.') from None
        result = dict(genres=[self.genre_box.get(index) for index in self.genre_box.curselection()],
                      min_rating=rating, language=self.language.get())
        if browsing:
            for name, variable in [('year_from', self.year_from), ('year_to', self.year_to)]:
                text = variable.get().strip()
                if text and (not text.isdigit() or not 1800 <= int(text) <= 2100):
                    raise ValueError('Enter a year from 1800 to 2100, or leave the year blank.')
                result[name] = int(text) if text else None
            if result['year_from'] and result['year_to'] and result['year_from'] > result['year_to']:
                raise ValueError('From year cannot be greater than to year.')
        return result

    def browse(self):
        self.clear('Browse movies', 'Search titles, combine filters, and select a row to see details. Language codes: en = English, hi = Hindi.')
        bar = ttk.Frame(self.page)
        bar.pack(fill='x', pady=(0, 10))
        self.search = tk.StringVar()
        entry = ttk.Entry(bar, textvariable=self.search)
        entry.pack(side='left', fill='x', expand=True, padx=(0, 10))
        entry.bind('<Return>', lambda event: self.apply_filters())
        self.sort = tk.StringVar(value='Rating')
        ttk.Combobox(bar, textvariable=self.sort, values=['Rating', 'Year', 'Title', 'Popularity'],
                     state='readonly', width=12).pack(side='left', padx=(0, 10))
        ttk.Button(bar, text='Search / Apply', command=self.apply_filters).pack(side='left')
        ttk.Button(bar, text='Clear Filters', command=self.browse).pack(side='left', padx=(8, 0))
        self.filters(self.page, browsing=True)
        self.movie_table(similar=True)
        self.populate(self.data.filter_movies())

    def apply_filters(self):
        try:
            options = self.read_filters(browsing=True)
            self.populate(self.data.filter_movies(search=self.search.get(), sort=self.sort.get(), **options))
        except ValueError as error:
            messagebox.showwarning('Check filters', str(error), parent=self.root)

    def movie_table(self, similar=False, remove=False):
        table_frame = ttk.Frame(self.page)
        table_frame.pack(fill='both', expand=True)
        columns = ('title', 'genres', 'rating', 'year', 'language', 'similarity')
        self.tree = ttk.Treeview(table_frame, columns=columns, show='headings', selectmode='browse', height=6)
        widths = [250, 180, 65, 65, 75, 100]
        for name, width in zip(columns, widths):
            self.tree.heading(name, text=name.title())
            self.tree.column(name, width=width, minwidth=60, stretch=(name in ['title', 'genres']))
        self.tree.grid(row=0, column=0, sticky='nsew')
        vertical = ttk.Scrollbar(table_frame, orient='vertical', command=self.tree.yview)
        vertical.grid(row=0, column=1, sticky='ns')
        horizontal = ttk.Scrollbar(table_frame, orient='horizontal', command=self.tree.xview)
        horizontal.grid(row=1, column=0, sticky='ew')
        self.tree.configure(yscrollcommand=vertical.set, xscrollcommand=horizontal.set)
        table_frame.rowconfigure(0, weight=1)
        table_frame.columnconfigure(0, weight=1)
        self.tree.bind('<<TreeviewSelect>>', self.on_select)
        self.details = ScrolledText(self.page, height=4, wrap='word', bg=PANEL, fg=TEXT,
                                   font=('Segoe UI', 10), relief='flat', padx=10, pady=8)
        self.details.pack(fill='x', pady=8)
        self.set_details('Select a movie to view its description and details.')
        buttons = ttk.Frame(self.page)
        buttons.pack(fill='x')
        ttk.Button(buttons, text='Add to Favorites', command=lambda: self.favorite_action(True)).pack(side='left')
        if remove:
            ttk.Button(buttons, text='Remove Favorite', command=lambda: self.favorite_action(False)).pack(side='left', padx=8)
        if similar:
            ttk.Button(buttons, text='Recommend Similar Movies', command=self.similar_movies).pack(side='left', padx=8)

    def set_details(self, text):
        self.details.configure(state='normal')
        self.details.delete('1.0', 'end')
        self.details.insert('1.0', text)
        self.details.configure(state='disabled')

    def populate(self, movies):
        self.selected_id = None
        self.tree.delete(*self.tree.get_children())
        self.set_details('Select a movie to view its description and details.')
        for row in movies.itertuples():
            score = getattr(row, 'similarity', None)
            self.tree.insert('', 'end', iid=str(row.id), values=(row.title, row.genre_text or 'Unknown',
                             f'{row.vote_average:.1f}', int(row.year) or 'Unknown', row.original_language,
                             '—' if score is None else f'{score * 100:.1f}%'))
        self.status.set(f'{len(movies):,} movies shown.' if len(movies) else 'No matching movies. Try broader genres, a lower rating, or another language.')

    def on_select(self, event=None):
        if self.tree is None or not self.tree.selection():
            return
        movie_id = int(self.tree.selection()[0])
        row = self.data.movie(movie_id)
        self.selected_id = movie_id
        self.set_details(f"{row['title']} | {row['genre_text'] or 'Unknown genre'}\n"
                         f"Language: {row['original_language']} | Year: {row['year'] or 'Unknown'} | "
                         f"Duration: {row['runtime']:g} min | Rating: {row['vote_average']:.1f}/10\n\n"
                         f"{row['overview'] or 'No description available.'}")
        try:
            self.data.view(movie_id)
            self.status.set(f"Viewing: {row['title']} • Added to recent history.")
        except OSError as error:
            self.status.set(f'Could not save history: {error}')

    def favorite_action(self, add):
        if self.selected_id is None:
            messagebox.showinfo('Select a movie', 'Select a movie row first.', parent=self.root)
            return
        try:
            self.data.set_favorite(self.selected_id, add)
            if not add:
                self.populate(self.data.by_ids(self.data.favorites))
            self.status.set('Favorite saved.' if add else 'Favorite removed.')
        except (OSError, ValueError) as error:
            messagebox.showerror('Could not save', str(error), parent=self.root)

    def recommendations(self):
        self.clear('AI recommendations', 'Choose genres or enter keywords. Language and rating are strict filters; similarity ranks the remaining movies.')
        self.filters(self.page)
        ttk.Button(self.page, text='Find My Top 5 Movies', command=self.preference_results).pack(anchor='w', pady=(0, 8))
        ttk.Label(self.page, text='Content similarity is not a probability that you will like a movie.', style='Muted.TLabel').pack(anchor='w', pady=(0, 8))
        self.movie_table()
        self.populate(self.data.movies.iloc[:0])
        self.status.set('Choose at least one genre or enter movie-related keywords.')

    def preference_results(self):
        try:
            self.populate(self.engine.preferences(keywords=self.keywords.get(), **self.read_filters()))
        except ValueError as error:
            messagebox.showwarning('Check preferences', str(error), parent=self.root)

    def similar_movies(self):
        if self.selected_id is None:
            messagebox.showinfo('Select a movie', 'Select a movie in Browse Movies first.', parent=self.root)
            return
        try:
            options = self.read_filters()
            movie_id = self.selected_id
            result = self.engine.similar(movie_id, options['language'], options['min_rating'])
        except ValueError as error:
            messagebox.showwarning('Check filters', str(error), parent=self.root)
            return
        title = self.data.movie(movie_id)['title']
        self.clear('Similar movies', f"Selected: {title}\nLanguage: {options['language']} • Minimum rating: {options['min_rating']:g}. Browse genres, years and search do not limit these results.")
        ttk.Button(self.page, text='Back to Browse Movies', command=self.browse).pack(anchor='w', pady=(0, 8))
        ttk.Label(self.page, text='Score = content similarity, not a guarantee of enjoyment.', style='Muted.TLabel').pack(anchor='w', pady=(0, 8))
        self.movie_table()
        self.populate(result)

    def favorites(self):
        self.clear('Your favorites', 'Your saved movie IDs are stored in data/favorites.json and reload at startup.')
        self.movie_table(remove=True)
        self.populate(self.data.by_ids(self.data.favorites))

    def history(self):
        self.clear('Recently viewed', 'Your last ten distinct movies, newest first. Selecting a movie moves it to the top on your next visit.')
        self.movie_table()
        self.populate(self.data.by_ids(self.data.history))

    def analytics(self):
        self.clear('Movie analytics', 'Explore the dataset. Ratings are the supplied dataset values, not verified measures of quality.')
        controls = ttk.Frame(self.page)
        controls.pack(fill='x', pady=(0, 8))
        self.chart_frame = ttk.Frame(self.page)
        self.chart_frame.pack(fill='both', expand=True)
        for label in ['Genres', 'Ratings', 'Top 10']:
            ttk.Button(controls, text=label, command=lambda kind=label: self.show_chart(kind)).pack(side='left', padx=(0, 10))
        self.show_chart('Genres')

    def show_chart(self, kind):
        if self.chart is not None:
            self.chart.get_tk_widget().destroy()
            self.chart.figure.clear()
        self.chart = FigureCanvasTkAgg(create_figure(self.data.movies, kind), master=self.chart_frame)
        self.chart.draw()
        self.chart.get_tk_widget().pack(fill='both', expand=True)
        self.status.set(f'{kind} chart • {len(self.data.movies):,} movies')

    def about(self):
        self.clear('About & help', 'An MCA mini-project demonstrating Python programming and content-based machine learning.')
        text = ScrolledText(self.page, wrap='word', bg=PANEL, fg=TEXT, font=('Segoe UI', 11),
                            relief='flat', padx=18, pady=15)
        text.pack(fill='both', expand=True)
        text.insert('1.0', '''OBJECTIVE
Help users discover movies with similar genres, descriptions and keywords, fully offline.

TECHNOLOGIES
Python, Tkinter/ttk, Pandas, NumPy, scikit-learn, Matplotlib, CSV and JSON.

SEARCH AND FILTER
Open Browse Movies, type all or part of a title, then press Enter or Search / Apply. Search ignores case. Click genres to toggle them; multiple genres use ANY-match filtering. Set rating, language or years, then apply. Clear Filters resets everything. Missing years are shown as Unknown and excluded when a year range is used. Language uses dataset codes (en = English, hi = Hindi, fr = French).

SIMILAR MOVIES
Select a row in Browse Movies, then click Recommend Similar Movies. Language and minimum rating apply to the results. The source movie is excluded. Up to five results appear in descending content similarity, then rating, then title. Zero similarity means no shared usable text features.

PERSONALIZED PREFERENCES
Open AI Recommendations. Choose at least one genre or enter keywords such as space adventure. Optionally select language and minimum rating, then Find My Top 5 Movies. Genres filter using ANY match. No positive text matches produces a helpful empty-result message.

TF-IDF IN SIMPLE WORDS
Text becomes a numerical vector. Common words such as “the” are removed. Words that distinguish a movie receive more importance than words appearing everywhere. We use genres, descriptions and dataset keywords, including one-word and two-word features.

COSINE SIMILARITY
Compare the direction of two text vectors. With these nonnegative TF-IDF vectors, the score ranges from 0 to 1. We display it as 0% to 100%. A high score means similar content, not a probability of enjoyment.

FAVORITES AND HISTORY
Select a row to view details and record it in history. Add to Favorites saves its ID. Remove Favorite is available on the Favorites page. The last ten distinct viewed IDs are saved in history.json. These lists do not train the model.

LIMITATIONS
This is content-based filtering, not collaborative filtering or a language model. Dataset coverage limits available languages and movies. Some highly rated movies have very few votes. No online posters, accounts, API keys or network requests are used.
''')
        text.configure(state='disabled')
```

### recommender.py
```python
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
```

### data_manager.py
```python
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
```

### analytics.py
```python
"""Create figures; the GUI embeds them using FigureCanvasTkAgg."""
from matplotlib.figure import Figure


def create_figure(movies, kind):
    figure = Figure(figsize=(9, 5), dpi=100, facecolor='#111c30', layout='constrained')
    axis = figure.add_subplot(111)
    axis.set_facecolor('#111c30')
    if kind == 'Genres':
        counts = movies['genre_list'].explode().dropna().value_counts().sort_values()
        axis.barh(counts.index, counts.values, color='#4d9fff')
        axis.set_title('Movie counts by genre (a movie can count more than once)')
        axis.set_xlabel('Movies')
    elif kind == 'Ratings':
        axis.hist(movies['vote_average'], bins=20, range=(0, 10), color='#4d9fff', edgecolor='#111c30')
        axis.set_title('Rating distribution — includes unrated / zero-rating entries')
        axis.set_xlabel('Dataset rating / 10')
        axis.set_ylabel('Movies')
    else:
        top = movies.sort_values(['vote_average', 'sort_title', 'id'], ascending=[False, True, True]).head(10).iloc[::-1]
        labels = [f'{row.title[:36]} ({row.id})' for row in top.itertuples()]
        axis.barh(labels, top['vote_average'], color='#4d9fff')
        axis.set_xlim(0, 10.5)
        axis.set_title('Ten highest dataset ratings — no minimum vote count')
        axis.set_xlabel('Rating / 10 (small vote counts can be misleading)')
    axis.tick_params(colors='#dbe7ff', labelsize=9)
    axis.title.set_color('white')
    axis.xaxis.label.set_color('#dbe7ff')
    axis.yaxis.label.set_color('#dbe7ff')
    for spine in axis.spines.values():
        spine.set_color('#344563')
    return figure
```

## 7. Windows and VS Code installation
1. Install Python 3.11 or 3.12 (64-bit recommended). Include Tcl/Tk support and select **Add Python to PATH** in the installer.
2. Extract the ZIP completely. Do not run it from inside the ZIP preview.
3. Open the extracted `movie_recommendation_system` folder in VS Code.
4. Open **Terminal → New Terminal**. The terminal must be inside the folder containing `main.py`.
5. Run these commands:
```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe main.py
```
Using the environment's Python directly avoids PowerShell activation-policy issues. If `py` is unavailable but `python` works, use `python -m venv .venv` for the first command.

Optional VS Code configuration: install Microsoft's Python extension, then use **Python: Select Interpreter** and select `.venv\Scripts\python.exe`. Running `main.py` starts the desktop app; Live Server is not needed.

## 8. Run the application
After setup, open a terminal in the project folder and run:
```powershell
.\.venv\Scripts\python.exe main.py
```
For an already configured environment:
```text
python main.py
```
Paths are relative to the source-file directory, so data loads correctly even when the working directory changes. Keep `data/movies.csv` inside the project. Favorites and history are written into the project’s `data` folder, which must be writable.

Troubleshooting:
- **No module named pandas/sklearn/matplotlib:** install requirements with the same interpreter used to run the app.
- **No module named tkinter:** modify/reinstall Python with Tcl/Tk support.
- **Dataset missing:** restore `data/movies.csv` from the ZIP.
- **Permission denied while saving:** extract the project into a writable location, such as Documents.
- **No matching movies:** lower the rating, choose All languages, broaden genres or use different keywords.
- **Empty or broken favorites/history JSON:** startup uses an empty list and displays a warning; a later successful save replaces that list.
- **Missing description:** details show “No description available”; genre/keyword features still work.
- Initial launch builds TF-IDF before showing the main window and can take a few seconds.
- Minimum supported window size is 960 × 650. Tables and details have scrollbars. On a small display with heavy Windows scaling, maximize the window or reduce display scaling.

## 9. Recommendation algorithm and example
### Preparing text
For each movie, concatenate genre names, the overview and keyword names. For example:
```text
Movie A: Science Fiction space crew explores an alien planet spaceship
Movie B: Science Fiction astronauts travel to a distant planet space mission
Movie C: Romance two people fall in love wedding
```
Movies A and B share more relevant content than A and C.

### TF-IDF
TF represents how often a term occurs in a movie's text. IDF reduces the influence of terms appearing in many movies. This project uses scikit-learn's smoothed IDF and sublinear TF:
```text
TF(t, d) = 1 + ln(count(t, d))     when the term is present; otherwise 0
IDF(t) = ln((1 + N) / (1 + df(t))) + 1
raw_weight(t, d) = TF(t, d) × IDF(t)
```
N is the number of movies and df(t) is the number containing term t. Vectors are L2-normalized. English stop words are removed; one-word and two-word features are included, with at most 20,000 features.

### Cosine similarity
```text
similarity(A, B) = dot(A, B) / (length(A) × length(B))
```
As a small vector illustration, A = [1, 1, 0], B = [1, 1, 1], C = [0, 0, 1]. Similarity(A, B) is about 0.816 and similarity(A, C) is 0. This illustration is not the actual fitted movie vectors.

For similar movies, the chosen movie supplies the query vector. For preferences, chosen genres and typed keywords supply the query text. Language and minimum rating are strict filters; preference genres also filter with ANY-match semantics. In Browse, genre/year/search filters limit the browsing table; only language and minimum rating carry over to similar-movie recommendations. The results screen states this explicitly.

Results sort by similarity descending, then rating descending, case-insensitive title ascending, and ID ascending for exact ties. The source movie is excluded. Similar-movie results can show zero-score candidates if necessary; preference recommendations require positive similarity. A result score of 21% means content similarity of 0.21, not a 21% chance of enjoying the movie.

This is a content-based machine-learning method. It neither learns from user ratings nor retrains from history. It can work without any user history, but only recommends movies already in the dataset. English stop words and mostly English descriptions also limit multilingual text quality.

## 10. Sample inputs and expected outputs
Measured against the included CSV and the installed dependency versions during development; minor numerical differences are possible with different library versions.

| Action | Input | Expected result |
|---|---|---|
| Dashboard | Launch | 4,803 movies; 20 genres; 85 rated at least 8 |
| Title search | `AVATAR` | Avatar; search ignores case |
| Similar movies | Avatar; All languages; rating 0 | Five results below; Avatar excluded |
| Preferences | Science Fiction; en; rating 7; `space adventure` | Five results below |
| Invalid rating | `hello`, `NaN`, or `11` | Clear warning; application remains open |
| Invalid range | From 2020, to 2000 | Clear warning |
| No keyword overlap | `zzzzzzzzzzzzzzzz` | No matching movies message |
| Favorite | Add Avatar twice, restart | Avatar appears once in Favorites |
| History | View 12 distinct movies | Only the newest 10 remain |

Actual Avatar similar-movie results:

| Movie | Rating | Content similarity |
|---|---:|---:|
| Aliens | 7.7 | 21.3% |
| Alien³ | 6.2 | 19.0% |
| Mission to Mars | 5.7 | 16.6% |
| Planet of the Apes | 5.6 | 16.3% |
| The Inhabited Island | 5.3 | 15.6% |

Actual preference results for Science Fiction + en + rating 7 + “space adventure”:

| Movie | Rating | Content similarity |
|---|---:|---:|
| Treasure Planet | 7.2 | 26.1% |
| Gattaca | 7.5 | 17.8% |
| Guardians of the Galaxy | 7.9 | 17.7% |
| Avatar | 7.2 | 16.1% |
| 2001: A Space Odyssey | 7.9 | 15.5% |

## 11. Feature testing checklist
Automated checks:
```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```
The tests verify the supplied dataset, JSON extraction, literal case-insensitive search, combined filters, source exclusion, deterministic similarity ordering, preference filters, empty results, tie-breakers, JSON persistence, ten-item history, corrupt history recovery, missing/invalid CSV errors and rendering all three chart figures. Tests use temporary folders for writes and leave your saved lists unchanged.

Manual desktop checklist:
- [ ] Open app; confirm dashboard figures and both shortcut buttons.
- [ ] Navigate every sidebar page, including History and About.
- [ ] Search partial titles, press Enter, then clear filters.
- [ ] Select two genres, rating, language and year bounds; apply.
- [ ] Sort by rating, title, year and popularity.
- [ ] Enter invalid rating/year values and verify helpful messages.
- [ ] Select a movie; check all details and history entry.
- [ ] Generate similar movies; verify source exclusion and score labels.
- [ ] Generate preference matches and exercise empty-result handling.
- [ ] Add favorites, restart, then remove a favorite.
- [ ] View more than ten movies and confirm history order on reopening History.
- [ ] Switch all three charts and inspect labels.
- [ ] Resize to the minimum window size and use scrollbars.
- [ ] Disconnect internet and confirm normal app operation.

Verification limits: backend tests and chart rendering were run in the build environment. That environment has no graphical display; live Tkinter button interaction and Windows-specific display layout still require the manual desktop checklist. No claim is made that the Windows GUI was visually tested here.

## 12. Ten viva questions and answers
1. **What kind of recommendation system is this?** Content-based filtering: it compares movie text features with a selected movie or preference query.
2. **What is TF-IDF?** A method for turning text into numerical vectors, giving more importance to distinguishing terms and less to terms common across documents.
3. **Why use cosine similarity?** It compares vector directions, helping measure shared content without simply favoring longer descriptions.
4. **Why use Pandas?** It reads the CSV and makes cleaning, filtering, sorting and aggregation convenient.
5. **Why must genres and keywords be parsed?** Their CSV cells contain JSON lists of dictionaries. We extract readable names before filtering and vectorization.
6. **How are favorites stored?** A JSON list of movie IDs is written to disk and reloaded at startup. IDs distinguish movies that might have the same title.
7. **How do you handle missing values?** Missing descriptions become empty text, invalid numeric values become zero, unknown dates display as Unknown, and missing titles/invalid IDs are skipped.
8. **Does the model learn from browsing?** No. History is only a saved list. Preference recommendations use the current explicit form selections.
9. **Why separate files?** GUI code, data storage, recommendation logic and charts have different responsibilities. Separation makes them easier to understand and change.
10. **What are the limitations?** Recommendations depend on the dataset and wording. No collaborative signals or feedback learning are included. Similarity is not proof of enjoyment, and high ratings can come from very few votes.

Syllabus mapping: DataManager and MovieRecommender demonstrate classes/objects; helper methods beginning with `_` demonstrate conventional encapsulation, not enforced privacy. Lists store IDs; sets combine genres; dictionaries map IDs to positions; tuples define GUI columns; loops build widgets and process records; functions handle events; Pandas and NumPy clean/filter data; scikit-learn builds vectors; Matplotlib draws charts. A while loop is deliberately not forced into the program.

## 13. Future improvements
- Add a minimum vote-count filter or a weighted rating to reduce small-sample rating effects.
- Add locally stored poster images and a poster grid.
- Support multiple local user profiles, each with separate JSON lists.
- Evaluate recommendation quality using a small human-reviewed relevance set.
- Add multilingual preprocessing if the dataset is expanded.
- Cache fitted vectors for faster startup, invalidating the cache when the CSV changes.
- Add explicit user feedback and evaluate any later personalization separately.

These are future ideas, not claims about features already implemented.
