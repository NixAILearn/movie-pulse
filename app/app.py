from pathlib import Path

import joblib
import pandas as pd
import streamlit as st
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity

HERE = Path(__file__).parent
MIN_VOTES = 20_000

st.set_page_config(page_title="Movie Pulse", page_icon="🎬", layout="wide")


@st.cache_data
def load_data():
    movies = pd.read_parquet(HERE / "data" / "movies_app.parquet")
    tags = pd.read_parquet(HERE / "data" / "tags_app.parquet")
    movies["label"] = (movies["title"] + " (" + movies["year"].astype("string") + ")"
                       + " · " + movies["director"].fillna("unknown"))
    return movies, tags

@st.cache_data
def load_tv():
    seasons = pd.read_parquet(HERE / "data" / "tv_seasons_app.parquet")
    sharks = pd.read_parquet(HERE / "data" / "tv_sharks_app.parquet")
    sharks = sharks.merge(
        seasons.groupby("showTconst", as_index=False)["showVotes"].first(), on="showTconst")
    sharks["label"] = sharks["showTitle"] + " (" + sharks["startYear"].astype("string") + ")"
    return seasons, sharks.sort_values("showVotes", ascending=False)

@st.cache_resource
def load_model():
    return joblib.load(HERE / "model.joblib")


@st.cache_resource
def build_matrix(_movies, _tags):
    row = pd.Index(_movies["tconst"]).get_indexer(_tags["tconst"])
    col, tag_names = pd.factorize(_tags["tag"])
    return csr_matrix((_tags["weight"].to_numpy(dtype=float), (row, col)),
                      shape=(len(_movies), len(tag_names)))


movies, tags = load_data()
model = load_model()
X = build_matrix(movies, tags)
FEATURES = list(model.feature_names_in_)
seasons, sharks = load_tv()
SHARK_DROP, SHARK_FIT = 0.5, 0.5

st.title("🎬 Movie Pulse")
st.caption(f"What makes a movie well-rated? {len(movies):,} movies with 1,000+ votes. "
           "Information courtesy of IMDb (https://www.imdb.com). Used with permission.")

explore, predict, like_this, tv, findings = st.tabs(
    ["Explore", "Predict", "Movies like this", "TV death curves", "Findings"])

# ---------- Explore ----------
with explore:
    all_genres = sorted(movies["genres"].str.split(",").explode().dropna().unique())
    c1, c2 = st.columns(2)
    lo, hi = int(movies["decade"].min()), int(movies["decade"].max())
    decades = c1.slider("Decades", lo, hi, (1950, hi), step=10)
    picked = c2.multiselect("Genres (empty = all)", all_genres)

    view = movies[movies["decade"].between(*decades)]
    if picked:
        has_genre = view["genres"].fillna("").str.split(",").apply(
            lambda gs: any(g in gs for g in picked))
        view = view[has_genre]

    m1, m2, m3 = st.columns(3)
    m1.metric("Movies", f"{len(view):,}")
    m2.metric("Average rating", f"{view['rating'].mean():.2f}" if len(view) else "–")
    m3.metric("Median votes", f"{view['votes'].median():,.0f}" if len(view) else "–")

    st.subheader("Average rating by decade")
    by_decade = view.groupby("decade")["rating"].mean().reset_index()
    by_decade["decade"] = by_decade["decade"].astype(str) + "s"
    st.bar_chart(by_decade, x="decade", y="rating", sort=False)

    st.subheader("Average rating by genre")
    by_genre = (view.assign(genre=view["genres"].str.split(","))
                    .explode("genre")
                    .groupby("genre")["rating"].agg(["mean", "size"])
                    .reset_index())
    st.bar_chart(by_genre[by_genre["size"] >= 50], x="genre", y="mean",
                 horizontal=True, sort="-mean")

    st.subheader("Runtime vs rating")
    sample = view[view["runtime"].between(60, 240)]
    st.scatter_chart(sample.sample(min(len(sample), 3000), random_state=42),
                     x="runtime", y="rating")

    # ---------- Predict ----------
with predict:
    st.write("Pick a movie. The model only sees facts known before release: "
             "year, runtime, genres and the director's earlier films.")
    tconst = st.selectbox("Movie", movies["tconst"], key="predict_movie",
                          format_func=dict(zip(movies["tconst"], movies["label"])).get)
    movie = movies[movies["tconst"] == tconst].iloc[0]

    feats = movie[["x_" + f for f in FEATURES]].to_frame().T.astype(float)
    feats.columns = FEATURES
    predicted = model.predict(feats)[0]

    a, b, c = st.columns(3)
    a.metric("Actual IMDb rating", f"{movie['rating']:.1f}")
    b.metric("Predicted", f"{predicted:.1f}", delta=f"{predicted - movie['rating']:+.1f}",
             delta_color="off")
    c.metric("Director's earlier films", int(movie["x_dir_prev_films"]))

    if movie["split"] == "train":
        st.warning("This film is from before 2015, so it was in the training data. "
                   "The model has seen its rating, so this is not a fair test.")
    else:
        st.success("This film is from 2015 or later: the model never saw it in training. "
                   "This is an honest test.")
    st.caption("Random forest, test MAE 0.73 vs 0.96 for always guessing the average.")

    # ---------- Movies like this ----------
with like_this:
    popular = movies[movies["votes"] >= MIN_VOTES]
    tconst = st.selectbox("Find movies like", popular["tconst"], key="like_movie",
                          format_func=dict(zip(popular["tconst"], popular["label"])).get)
    i = movies.index[movies["tconst"] == tconst][0]

    scores = cosine_similarity(X[i], X).ravel()
    recs = (movies.assign(score=scores)
                  .drop(index=i)
                  .query("votes >= @MIN_VOTES")
                  .sort_values(["score", "votes"], ascending=False)
                  .head(5))
    st.dataframe(recs[["title", "year", "director", "genres", "rating", "score"]],
                 hide_index=True, width="stretch",
                 column_config={"score": st.column_config.ProgressColumn(
                     "similarity", min_value=0, max_value=1, format="%.2f")})

    other = st.selectbox("Why? Compare with", recs["tconst"],
                         format_func=dict(zip(recs["tconst"], recs["label"])).get)
    ta = tags[tags["tconst"] == tconst]
    tb = tags[tags["tconst"] == other]
    shared = ta[ta["tag"].isin(tb["tag"])].sort_values("weight", ascending=False)
    st.dataframe(shared[["kind", "label", "n_films", "weight"]].round(2), hide_index=True)


# ---------- TV death curves ----------
with tv:
    st.write("Pick a long-running show. The detector splits its seasons into a "
             "\"before\" and \"after\" level and finds where the shift happened.")
    show_id = st.selectbox("Show", sharks["showTconst"], key="tv_show",
                           format_func=dict(zip(sharks["showTconst"], sharks["label"])).get)
    show = sharks[sharks["showTconst"] == show_id].iloc[0]
    g = seasons[seasons["showTconst"] == show_id].sort_values("seasonNumber")

    a, b, c, d = st.columns(4)
    a.metric("Shark season", f"S{show['shark_season']}")
    b.metric("Before", f"{show['before']:.2f}")
    c.metric("After", f"{show['after']:.2f}", delta=f"{-show['drop']:+.2f}")
    d.metric("Fit gain", f"{show['fit_gain']:.2f}")

    if show["drop"] >= SHARK_DROP and show["fit_gain"] >= SHARK_FIT:
        st.error(f"Jumped the shark in season {show['shark_season']}.")
    elif show["drop"] <= -SHARK_DROP and show["fit_gain"] >= SHARK_FIT:
        st.success(f"Got better from season {show['shark_season']} on.")
    elif show["drop"] >= SHARK_DROP:
        st.warning(f"A big drop from season {show['shark_season']}, but the seasons bounce "
                   "around too much to call it one clean shift.")
    else:
        st.info("No clear shift: the split doesn't beat one flat line by enough.")

    chart = pd.DataFrame({
        "season": g["seasonNumber"].astype(int),
        "season average": g["avg_rating"].round(2),
        "before / after level": [show["before"] if s < show["shark_season"] else show["after"]
                                 for s in g["seasonNumber"]],
    })
    st.line_chart(chart, x="season", y=["season average", "before / after level"])

    st.subheader("Biggest sharks")
    jumped = (sharks[(sharks["drop"] >= SHARK_DROP) & (sharks["fit_gain"] >= SHARK_FIT)]
              .sort_values("drop", ascending=False))
    st.dataframe(jumped[["showTitle", "startYear", "n_seasons", "shark_season",
                         "before", "after", "drop"]].head(15),
                 hide_index=True, width="stretch")
    st.caption(f"{len(jumped)} of {len(sharks)} long-running shows jumped: a drop of at least "
               f"{SHARK_DROP} and a fit gain of at least {SHARK_FIT}.")

    # ---------- Findings ----------
with findings:
    st.markdown("""
### What I found
1. **Older films rate higher, but that's survivorship bias.** Only old films people still watch
   get 1,000+ votes today. 
2. **Genre matters.** Horror is lowest on average; Documentary and Biography are highest. Was not expecting the Thriller genre to in the bottom 3 genres
3. **Consistent isn't the same as great. Surprised to see no familiar names in consistent directors but their Average rating is high for e.g Nolan with average rating of 8.18 but was not in top 5 consistent directors** 

### How well can a model predict a rating?
- Using only facts known before release, a random forest is off by **0.73** stars on average
  for films from 2015 on, vs **0.96** for always guessing the average: 24% better.
- The director's past average is the strongest feature. It also pulls predictions toward
  the middle, so great films get under-predicted.
- A director average that includes the film itself scored a fake 0.31. That's leakage.

### Limits
- No plot, budget or box office in IMDb's free data.
- The recommender only knows genres, decade, director and top 3 cast.
- Ratings are from IMDb users, who aren't everyone.
""")