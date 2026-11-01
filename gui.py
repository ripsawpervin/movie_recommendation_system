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
