import streamlit as st
import pandas as pd
import joblib

#  Platform URLs
platform_links = {
    "Netflix": "https://www.netflix.com",
    "Hulu": "https://www.hulu.com",
    "Prime Video": "https://www.primevideo.com",
    "Disney+": "https://www.disneyplus.com"
}

#  Load model and encoders
@st.cache_data
def load_model():
    model = joblib.load("mood_to_genre_model.pkl")
    le = joblib.load("mood_label_encoder.pkl")
    mlb = joblib.load("genre_binarizer.pkl")
    return model, le, mlb

#  Load movie data
@st.cache_data
def load_data():
    df = pd.read_csv("cleaned_data.csv")
    return df

#  Recommendation logic
def recommend_movies_by_mood(mood, model, le, mlb, df):
    try:
        mood_encoded = le.transform([mood])
        predicted = model.predict(mood_encoded.reshape(-1, 1))
        genres = mlb.inverse_transform(predicted)[0]
    except:
        return [], genres

    def match_genres(genre_str):
        if pd.isna(genre_str): return False
        genre_list = [g.strip() for g in genre_str.split(",")]
        return any(g in genre_list for g in genres)

    filtered = df[df["Genre"].apply(match_genres)]

    def has_platform(row):
        return any(str(row.get(p, 0)) == "1" for p in platform_links.keys())

    filtered = filtered[filtered.apply(has_platform, axis=1)]
    if "IMDb" in filtered.columns:
        filtered = filtered.dropna(subset=["IMDb"])
        filtered = filtered.sort_values(by="IMDb", ascending=False).head(10)
    netflix_matches = filtered[filtered["Netflix"] == 1]
    st.write(" Netflix matches after genre filter:", netflix_matches.shape[0])
# If filtered is empty, return fallback movies
    if filtered.empty:
       fallback = df[df.apply(has_platform, axis=1)]
       fallback = fallback.dropna(subset=["IMDb"])
       fallback = fallback.sort_values(by="IMDb", ascending=False).head(10)
       return fallback, ["Fallback (top-rated available movies)"]
    else:
       return filtered.head(10), genres
#  Streamlit UI
st.title("MoodFlix ")
st.write("Select your current mood, and we'll recommend movies to change it ")

model, le, mlb = load_model()
df = load_data()

# Let user select a mood
available_moods = list(le.classes_)
selected_mood = st.selectbox("Choose your mood:", available_moods)


# Recommend movies
if st.button("Get Recommendations"):
    movies, genres = recommend_movies_by_mood(selected_mood, model, le, mlb, df)

    if not movies.empty:
        st.success(" Here are the top movies to lift your mood!")

        for _, row in movies.iterrows():
            st.subheader(row["Title"])
            st.write(f" IMDb: {row['IMDb']}")
            st.write(f" Genre: {row['Genre']}")

            platforms = []
            for p, link in platform_links.items():
                if str(row.get(p, 0)) == "1":
                    platforms.append(f"[{p}]({link})")

            st.markdown(f" Available on: {' | '.join(platforms)}")
            st.markdown("---")
    else:
        st.warning("No matching movies with known platforms found for this mood.")
