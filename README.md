# Movie Pulse

What makes a movie well-rated, and can a model predict it?
A learning project in data science and ML using IMDb's official non-commercial datasets.

## Data
Information courtesy of IMDb (https://www.imdb.com). Used with permission.
Raw files come from https://datasets.imdbws.com/ and are not stored in this repo.
To reproduce, download them into data/raw/ (see Sprint 1).

## Progress log
- **Sprint 1: Setup and first look.** Set up the repo, .venv and .gitignore; downloaded IMDb's five official datasets into data/raw/ (not committed).
- Loaded title.basics (millions of rows) with usecols, dtypes and category columns to keep memory down; treated \N as missing.
- Found that TV episodes dominate the file and movies are a small slice, so every later analysis filters to movies first.