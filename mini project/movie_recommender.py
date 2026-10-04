"""
Minor Project: Movie Recommendation System (MovieLens 100k)
Content-based + Collaborative filtering (SVD) + Hybrid

Setup:
    pip install pandas numpy scikit-learn scikit-surprise matplotlib
Data:
    Download https://www.kaggle.com/datasets/grouplens/movielens-100k-dataset
    Unzip so that DATA_DIR contains u.data and u.item
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from surprise import Dataset, Reader, SVD, accuracy
from surprise.model_selection import train_test_split

DATA_DIR = "ml-100k"  # change if needed

# ---------- 1-2. Define type + collect data ----------
GENRES = ["unknown", "Action", "Adventure", "Animation", "Children", "Comedy",
          "Crime", "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror",
          "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western"]

ratings = pd.read_csv(f"{DATA_DIR}/u.data", sep="\t",
                      names=["user", "item", "rating", "ts"])
movies = pd.read_csv(f"{DATA_DIR}/u.item", sep="|", encoding="latin-1",
                     names=["item", "title", "date", "vdate", "url"] + GENRES)

# ---------- 3. Preprocess ----------
ratings = ratings.dropna().drop_duplicates(["user", "item"])
movies = movies.drop(columns=["vdate", "url"]).dropna(subset=["title"])
movies = movies.set_index("item")
genre_matrix = movies[GENRES].values.astype(float)
# L2-normalise so cosine similarity is well-behaved
norms = np.linalg.norm(genre_matrix, axis=1, keepdims=True)
genre_matrix = genre_matrix / np.where(norms == 0, 1, norms)

# ---------- 4. EDA ----------
print(ratings.describe())
print("Sparsity: %.2f%%" % (100 * (1 - len(ratings) / (ratings.user.nunique() * ratings.item.nunique()))))
ratings.rating.value_counts().sort_index().plot(kind="bar", title="Rating distribution")
plt.savefig("rating_distribution.png", bbox_inches="tight")
plt.close()

# ---------- 5. Model user behaviour (similarity) ----------
item_ids = movies.index.to_list()
item_pos = {iid: i for i, iid in enumerate(item_ids)}
item_sim = cosine_similarity(genre_matrix)  # content similarity between movies


def similar_movies(title_part, n=10):
    match = movies[movies.title.str.contains(title_part, case=False, regex=False)]
    if match.empty:
        return "Movie not found"
    i = item_pos[match.index[0]]
    top = np.argsort(-item_sim[i])[1:n + 1]
    return movies.iloc[top].title.tolist()


# ---------- 6. Build engine (collaborative filtering) ----------
data = Dataset.load_from_df(ratings[["user", "item", "rating"]], Reader(rating_scale=(1, 5)))
trainset, testset = train_test_split(data, test_size=0.2, random_state=42)
svd = SVD(n_factors=100, n_epochs=30, random_state=42)
svd.fit(trainset)

# ---------- 8. Evaluate ----------
preds = svd.test(testset)
print("RMSE:", accuracy.rmse(preds))
print("MAE :", accuracy.mae(preds))


# ---------- 7. Generate recommendations ----------
def recommend_cf(user, n=10):
    seen = set(ratings[ratings.user == user].item)
    cand = [i for i in item_ids if i not in seen]
    scored = [(i, svd.predict(user, i).est) for i in cand]
    scored.sort(key=lambda x: -x[1])
    return scored[:n]


def recommend_hybrid(user, n=10, w_cf=0.7):
    """Hybrid = w * CF score (scaled 0-1) + (1-w) * content match to liked movies."""
    user_r = ratings[ratings.user == user]
    liked = user_r[user_r.rating >= 4].item
    seen = set(user_r.item)
    if liked.empty:
        return recommend_cf(user, n)
    profile = genre_matrix[[item_pos[i] for i in liked if i in item_pos]].mean(axis=0, keepdims=True)
    content = cosine_similarity(genre_matrix, profile).ravel()
    out = []
    for i in item_ids:
        if i in seen:
            continue
        cf = svd.predict(user, i).est / 5.0
        out.append((i, w_cf * cf + (1 - w_cf) * content[item_pos[i]]))
    out.sort(key=lambda x: -x[1])
    return out[:n]


def show(recs, label):
    print(f"\n{label}")
    for iid, score in recs:
        print(f"  {movies.loc[iid, 'title']:<55} {score:.3f}")


if __name__ == "__main__":
    print("\nSimilar to 'Toy Story':", similar_movies("Toy Story"))
    show(recommend_cf(196), "Collaborative (user 196)")
    show(recommend_hybrid(196), "Hybrid (user 196)")