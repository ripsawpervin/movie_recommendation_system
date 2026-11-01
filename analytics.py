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
