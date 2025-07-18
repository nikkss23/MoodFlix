import streamlit as st
import pandas as pd
import joblib

# Platform URLs
platform_links = {
    "Netflix": "https://www.netflix.com",
    "Hulu": "https://www.hulu.com",
    "Prime Video": "https://www.primevideo.com",
    "Disney+": "https://www.disneyplus.com"
}

# Load model and encoders
@st.cache_data
def load_model():
    model = joblib.load("mood_to_genre_model.pkl")
    le = joblib.load("mood_label_encoder.pkl")
    mlb = joblib.load("genre_binarizer.pkl")
    return model, le, mlb

# Load movie data
@st.cache_data
def load_data():
    df = pd.read_csv("cleaned_data.csv")
    return df

# Recommend movies based on mood
def recommend_movies_by_mood(mood, model, le, mlb, df):
    try:
        mood_encoded = le.transform([mood])
        predicted = model.predict(mood_encoded.reshape(-1, 1))
        genres = mlb.inverse_transform(predicted)[0]
    except:
        return [], []

    def match_genres(genre_str):
        if pd.isna(genre_str): return False
        genre_list = [g.strip().lower() for g in genre_str.split(",")]
        return any(pred.lower() in genre_list for pred in genres)

    filtered = df[df["Genre"].apply(match_genres)]

    def has_platform(row):
        return any(row.get(p, 0) in [1, "1"] for p in platform_links.keys())

    filtered = filtered[filtered.apply(has_platform, axis=1)]

    # Ensure platform columns exist
    platform_columns = list(platform_links.keys())
    missing_cols = [col for col in platform_columns if col not in filtered.columns]
    for col in missing_cols:
        filtered[col] = 0

    if "IMDb" in filtered.columns:
        filtered = filtered.dropna(subset=["IMDb"])
        filtered = filtered.sort_values(by="IMDb", ascending=False).head(10)

    # Fallback logic if no genre-matching movies found
    if filtered.empty:
        if "Netflix" in df.columns:
            netflix_fallback = df[df["Netflix"] == 1]
            netflix_fallback = netflix_fallback.dropna(subset=["IMDb"])
            netflix_fallback = netflix_fallback.sort_values(by="IMDb", ascending=False).head(10)
            if not netflix_fallback.empty:
                return netflix_fallback, ["Top Netflix Picks"]

        fallback = df[df.apply(has_platform, axis=1)]
        fallback = fallback.dropna(subset=["IMDb"])

        for col in platform_columns:
            if col not in fallback.columns:
                fallback[col] = 0

        fallback = fallback[fallback[platform_columns].sum(axis=1) > 0]
        fallback = fallback.sort_values(by="IMDb", ascending=False).head(10)
        return fallback, ["Fallback (top-rated platform movies)"]

    return filtered.head(10), genres

# Streamlit App UI
st.title("MoodFlix")
st.write("Select your current mood, and we'll recommend movies to change it")

# Load models and data
model, le, mlb = load_model()
df = load_data()

# Mood selection
available_moods = list(le.classes_)
selected_mood = st.selectbox("Choose your mood:", available_moods)

# Recommendation button
if st.button("Get Recommendations"):
    movies, genres = recommend_movies_by_mood(selected_mood, model, le, mlb, df)

    if not movies.empty:
        st.success("Here are the top movies to lift your mood!")

        for _, row in movies.iterrows():
            st.subheader(row["Title"])
            st.write(f"IMDb: {row['IMDb']}")
            st.write(f"Genre: {row['Genre']}")

            platforms = []
            for p, link in platform_links.items():
                if row.get(p, 0) in [1, "1"]:
                    platforms.append(f"[{p}]({link})")

            if platforms:
                st.markdown(f"Available on: {' | '.join(platforms)}")
            else:
                st.markdown("Platform unknown")

            st.markdown("---")
    else:
        st.warning("No matching movies found.")
