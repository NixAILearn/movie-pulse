# Movie Pulse

What makes a movie well-rated, and can a model predict it?
**Live app:** https://movie-pulse.streamlit.app
A learning project in data science and ML using IMDb's official non-commercial datasets.

## Data
Information courtesy of IMDb (https://www.imdb.com). Used with permission.
Raw files come from https://datasets.imdbws.com/ and are not stored in this repo.
To reproduce, download them into data/raw/ (see Sprint 1).

## Progress log
- **Sprint 1: Setup and first look.** Set up the repo, .venv and .gitignore; downloaded IMDb's five official datasets into data/raw/ (not committed).
- Loaded title.basics (millions of rows) with usecols, dtypes and category columns to keep memory down; treated \N as missing.
- Found that TV episodes dominate the file and movies are a small slice, so every later analysis filters to movies first.
- **Sprint 2: Clean movie table.** Filtered IMDb to non-adult movies, converted years and runtimes to numbers, and joined ratings (kept movies with 1,000+ votes).
- Added each movie's main director with two lookup joins (title.crew, name.basics), checking keys with validate, and one-hot encoded genres.
- Saved 49453 movies to data/processed/movies.parquet, with a cleaning log of rows kept at each step and the missing-data choices written down.
- **Sprint 3: Exploration and findings.** Explored 49,453 movies: ratings by decade, genre, runtime and votes, plus the most consistent directors.
- Learned groupby, explode, scatter plots, log scales and correlation; set minimum group sizes to avoid small-sample rankings.
- Found a difference between consistent diretors vs highly rated directors that I was expecting to see , and saw survivorship bias, confounders and a leakage risk votes to avoid in Sprint 4.
- **Sprint 4: Rating models.** Predicted IMDb rating from pre-release facts only (year, runtime, genres, director's past average), training on films before 2015 and testing on 2015+.
- Baseline MAE 0.961; linear regression 0.755; random forest 0.728 (24% better). A default deep forest overfit (train 0.25, test 0.77) until min_samples_leaf=20.
- Leakage check: a director average that includes the film itself scored a fake 0.313. Biggest misses are review-bombed films and breakout hits the features can't see.
- **Sprint 5: Recommender.** Built a "movies like this" recommender: each movie is a sparse vector of tags (genres, decade, director, top 3 cast), compared with cosine similarity.
- Version 1 (equal weights) ranked obscure films first, because thin profiles of 3-4 tags match too easily. IDF weighting (rare tags count more), double weight on genres and a 20,000-vote floor fixed it: Parasite now returns five Bong Joon Ho films.
- No answer key, so I sanity-checked by eye and built explain() to show shared tags. Limits: no plot data (Oppenheimer gets documentaries), and the vote floor adds popularity bias.