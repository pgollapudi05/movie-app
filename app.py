# app.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from scipy.sparse import csr_matrix
from sklearn.preprocessing import normalize

# ---------- 1) LOAD DATA ----------
@st.cache_data
def load_data():
    movies = pd.read_csv("data/ml-latest-small/movies.csv")
    ratings = pd.read_csv("data/ml-latest-small/ratings.csv")
    # Basic cleanups: genres split-for-display, year extraction
    movies["year"] = movies["title"].str.extract(r'\((\d{4})\)').astype("float").astype("Int64")
    movies["clean_title"] = movies["title"].str.replace(r'\s*\(\d{4}\)$', '', regex=True)
    return movies, ratings

movies, ratings = load_data()

st.set_page_config(page_title="Movie Ratings Explorer & Recommender", layout="wide")
st.title("🎬 Movie Ratings Explorer & Recommender")
with st.expander("ℹ️ About this app", expanded=True):
    st.write("""
    This is a mini *Movie Ratings Explorer & Recommender* built with Python and Streamlit.  
    It uses the [MovieLens latest-small dataset](https://grouplens.org/datasets/movielens/) (100k ratings across 9,000+ movies by 600+ users).

    *Features:*
    - 📊 Explore ratings distribution, most-rated movies, and top-rated movies.
    - 🔍 Find movies similar to a given title using *cosine similarity* on user–movie ratings.
    - 🎯 Generate personalized recommendations for any user ID in the dataset.

    *Tech stack:* Python · Streamlit · Pandas · NumPy · SciPy · scikit-learn · Plotly  
    """)

# Sidebar navigation
mode = st.sidebar.radio("Choose a mode:", ["Explore data", "Find similar movies", "Recommend for a user"])

# ---------- 2) QUICK STATS & EDA ----------
def show_explorer():
    st.subheader("Dataset overview")
    col1, col2, col3 = st.columns(3)
    col1.metric("Unique users", ratings["userId"].nunique())
    col2.metric("Unique movies", movies["movieId"].nunique())
    col3.metric("Total ratings", len(ratings))

    st.markdown("#### Ratings distribution")
    fig = px.histogram(ratings, x="rating", nbins=10, title="Distribution of ratings (1–5 stars)")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### Most-rated movies")
    counts = ratings.groupby("movieId")["rating"].count().rename("rating_count")
    avg = ratings.groupby("movieId")["rating"].mean().rename("avg_rating")
    agg = pd.concat([counts, avg], axis=1).reset_index().merge(movies[["movieId","clean_title","year"]], on="movieId", how="left")
    min_ratings = st.slider("Minimum number of ratings to include", 10, 200, 50, 10)
    top = agg[agg["rating_count"] >= min_ratings].sort_values("rating_count", ascending=False).head(30)
    fig2 = px.bar(top, x="clean_title", y="rating_count", hover_data=["avg_rating","year"], title="Top movies by #ratings")
    fig2.update_layout(xaxis_title="", xaxis_tickangle=45)
    st.plotly_chart(fig2, use_container_width=True)

    st.markdown("#### Best-rated (with minimum #ratings)")
    best = agg[agg["rating_count"] >= min_ratings].sort_values("avg_rating", ascending=False).head(30)
    fig3 = px.bar(best, x="clean_title", y="avg_rating", hover_data=["rating_count","year"], title="Top movies by average rating")
    fig3.update_layout(xaxis_title="", xaxis_tickangle=45, yaxis_range=[3.5,5])
    st.plotly_chart(fig3, use_container_width=True)

# ---------- 3) ITEM-ITEM SIMILARITY (COSINE) ----------
# We’ll compute similarities on the fly (efficient for this small dataset).
@st.cache_resource
def build_item_matrix():
    # Pivot to users x movies matrix (ratings, 0 if missing)
    user_item = ratings.pivot_table(index="userId", columns="movieId", values="rating").fillna(0.0)
    # Sparse matrix (users x movies)
    R = csr_matrix(user_item.values)
    # L2-normalize columns so cosine similarity becomes dot product
    R_norm = normalize(R, axis=0)
    return user_item, R_norm

user_item, R_norm = build_item_matrix()

def similar_movies(movie_id, top_n=10):
    # Find column index for this movie
    if movie_id not in user_item.columns:
        return []
    col_idx = user_item.columns.get_loc(movie_id)
    target_vec = R_norm[:, col_idx]                    # (users x 1)
    sims = R_norm.T @ target_vec                       # (movies x 1) sparse x sparse -> sparse
    sims = sims.toarray().ravel()
    sims[col_idx] = -1.0                               # exclude itself
    top_idx = sims.argsort()[-top_n:][::-1]            # top N
    top_movie_ids = user_item.columns[top_idx]
    scores = sims[top_idx]
    out = pd.DataFrame({"movieId": top_movie_ids, "similarity": scores})
    out = out.merge(movies[["movieId", "clean_title", "year"]], on="movieId", how="left")
    return out

def recommend_for_user(user_id, top_n=10, like_threshold=4.0, fallback_k=5):
    # If user not found, return empty
    if user_id not in user_item.index:
        return pd.DataFrame(columns=["movieId","score","clean_title","year"])

    user_row = user_item.loc[user_id]
    liked = user_row[user_row >= like_threshold]
    if liked.empty:
        # fallback: use user’s top-rated few movies
        liked = user_row.sort_values(ascending=False).head(fallback_k)

    scores = np.zeros(user_item.shape[1])
    # Add up similarities weighted by the user's ratings
    for movie_id, rating in liked.items():
        col_idx = user_item.columns.get_loc(movie_id)
        sims = (R_norm.T @ R_norm[:, col_idx]).toarray().ravel()
        scores += rating * sims

    # Remove already-rated movies
    already = user_row[user_row > 0].index
    already_idx = [user_item.columns.get_loc(mid) for mid in already]
    scores[already_idx] = -np.inf

    # Top N
    top_idx = np.argpartition(scores, -top_n)[-top_n:]
    top_idx = top_idx[np.argsort(scores[top_idx])[::-1]]

    out = pd.DataFrame({
        "movieId": user_item.columns[top_idx],
        "score": scores[top_idx]
    })
    out = out.merge(movies[["movieId","clean_title","year"]], on="movieId", how="left")
    return out

# ---------- 4) UI FLOW ----------
if mode == "Explore data":
    show_explorer()

elif mode == "Find similar movies":
    st.subheader("Find movies similar to a title you like")
    # Choose a reference movie
    movie_choice = st.selectbox(
        "Pick a movie:",
        options=movies.sort_values("clean_title")["clean_title"],
        index=0
    )
    # Map back to movieId
    picked = movies.loc[movies["clean_title"] == movie_choice, "movieId"].values[0]
    top = similar_movies(picked, top_n=10)
    st.write(f"Movies most similar to **{movie_choice}**:")
    st.dataframe(top[["clean_title","year","similarity"]].reset_index(drop=True))

elif mode == "Recommend for a user":
    st.subheader("Personalized recommendations for a given user ID")
    uid = st.number_input("Enter a userId (try numbers between 1 and 610)", min_value=1, step=1, value=1)
    recs = recommend_for_user(uid, top_n=10)
    if recs.empty:
        st.info("User not found in this dataset. Try a different number.")
    else:
        st.dataframe(recs[["clean_title","year","score"]].reset_index(drop=True))
        st.caption("Tip: If the results look odd, try a different userId—some users have rated very few movies.")
