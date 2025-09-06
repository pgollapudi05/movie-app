# Movie Ratings Explorer & Recommender

A Streamlit web app that explores the MovieLens *latest-small* dataset (~100k ratings) and recommends movies using *item–item cosine similarity*.
[View Live](https://movie-app-recommender.streamlit.app)
## Features
- 📊 Ratings distribution, most-rated and top-rated movies
- ⁠⁠🔍 “Similar to this title” using cosine similarity
- ⁠🎯 User-based recommendations for any ⁠ userId ⁠ in the dataset

## Stack
Python · Streamlit · Pandas · NumPy · SciPy · scikit-learn · Plotly

## Run locally
```bash
python -m venv .venv
# Activate: source .venv/bin/activate  (macOS/Linux)
#          .venv\Scripts\activate      (Windows)

python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
