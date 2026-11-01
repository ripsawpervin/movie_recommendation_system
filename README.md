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
All five Python source files are included beside this README, with short comments and meaningful names. **COMPLETE_PROJECT_GUIDE.md** contains this documentation with the full source of each Python file inserted in this section, in addition to the runnable files themselves.

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
