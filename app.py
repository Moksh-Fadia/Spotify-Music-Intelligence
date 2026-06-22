import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder      # Standar Scaler standardizes the features by removing the mean and scaling to unit variance ie. normalize the data, Label Encoder is used to convert categorical labels (words) into numerical form for ML models to understand (eg. "rock" -> 0, "pop" -> 1)
from sklearn.model_selection import train_test_split, cross_val_score  
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.cluster import KMeans
from xgboost import XGBClassifier
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Spotify ML Project", layout="wide")   
st.title("Spotify Music Analytics & Recommendation System")

spotify_df = pd.read_csv("spotify.csv")

playlist_files = {      # Mapping of playlist names to their corresponding CSV files as they will be used in the Mood-Based Clustering section
    "High-Energy Electronic": "csvs/high-energy_electronic.csv",
    "Chill Indie": "csvs/chill_indie.csv",
    "Slow Sad Acoustic": "csvs/slow_sad_acoustic.csv",
    "Danceable Pop Vibes": "csvs/danceable_pop_vibes.csv"
}

# --- SIDEBAR NAVIGATION ---
st.sidebar.title("Navigation")

section = st.sidebar.radio("Choose Feature", [  
    "Song Recommender",
    "Genre Prediction",
    "Hit/Flop Prediction",
    "Mood Based Clustering"
])

def normalize(text):
    return ' '.join(text.lower().strip().split())

def scale_features(df, features):       # This function takes a DataFrame and a list of feature names, applies standard scaling to those features, and returns the scaled features
    scaler = StandardScaler()    
    return scaler.fit_transform(df[features])    # The fit_transform method computes the mean and standard deviation for scaling and then applies the transformation (simplified values/numbers) to the specified features, returning a NumPy array of scaled values.


# --- SECTION 1: GENRE CLASSIFICATION ---
if section == "Genre Prediction":    # genres like rock, pop, hip-hop, etc.
    st.header("🎵 Genre Prediction")
    genre_df = spotify_df.copy()
    genre_df.dropna(inplace=True)

    X = genre_df.drop(columns=['track_id', 'artists', 'album_name', 'track_name', 'track_genre'])   # remove non-numeric/unnecessary columns and the target variable (track_genre) from the feature set [target variable removed to prevent data leakage]

    encoder = LabelEncoder()
    y = encoder.fit_transform(genre_df['track_genre'])      # ['pop', 'rock', 'edm'] = [0, 1, 2]
  
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
# stratify=y ensures that the train and test sets have the same proportion of each genre as the original dataset, which is important for imbalanced datasets
# random_state=42 ensures reproducibility of the train-test split meaning that every time the code is run, the same split will be produced to ensures model evaluation is consistent throughout

    xgb = XGBClassifier(    
        objective='multi:softmax',    # multi-class classification problem where the model predicts the class with the highest probability
        num_class=len(np.unique(y)),    # number of unique genres in the target variable
        eval_metric='mlogloss',     # mlogloss is a standard evaluation metric for multi-class classification. XGBoost uses it during training to measure prediction error
        random_state=42
    )    

    xgb.fit(X_train, y_train)
    y_pred = xgb.predict(X_test)

       # --- Genre Prediction ---
    st.subheader("🎯 Predict Song Genre")
    song_input = st.text_input("Enter Song Name:")

    if song_input:
        match = genre_df[genre_df['track_name'].str.lower() == song_input.lower()]  
        if match.empty:
            st.error("Song not found.")
        else:
            song_row = match.iloc[0]    # This line selects the first row of the match DataFrame, which contains the details of the song that matches the user input. It is used to extract the features of that specific song for genre prediction.
            song_features = song_row[X.columns].values.reshape(1, -1)    # This line extracts the feature values of the matched song (excluding non-numeric columns) and reshapes them into a 2D array with one row and multiple columns. This format is required for making predictions with the trained XGBoost model.
            pred = xgb.predict(song_features)   # song_features looks like [[0.5, 0.7, 0.3, ...]] and the model predicts the genre class index (e.g., 0, 1, 2) for that song based on its features.
            predicted_genre = encoder.inverse_transform(pred)[0]    # This line converts the predicted genre class index back to its original string label (e.g., "pop", "rock") using the inverse_transform method of the LabelEncoder. The [0] at the end retrieves the first (and only) element from the resulting array, giving us the predicted genre as a string. eg: 5 --> "pop"

            st.markdown(f"**{song_row['track_name']}** by *{song_row['artists']}*")
            st.success(f"Predicted Genre: **{predicted_genre}**")

    st.markdown("---")

    st.subheader("📊 Model Evaluation")
    st.subheader("Classification Report (Train/Test Split)")
    st.code(classification_report(y_test, y_pred))   # The classification report provides a detailed breakdown of the model's performance on the test set, including precision, recall, F1-score, and support for each genre class. This helps in understanding how well the model is performing across different genres.

    scores = cross_val_score(xgb, X, y, cv=5, scoring='f1_weighted')    # cross_val_score performs K-Fold Cross Validation on the entire dataset (X, y) using the XGBoost classifier (xgb). It splits the data into 5 folds (cv=5), trains the model on 4 folds, and tests it on the remaining fold. This process is repeated 5 times, with each fold serving as the test set once. The scoring parameter 'f1_weighted' calculates the weighted average F1 score across all classes, which accounts for class imbalance by giving more weight to classes with more samples.

    st.subheader("K-Fold Cross Validation (F1 Weighted)")
    st.write("Scores:", scores)
    st.write("Average F1 Score:", np.round(scores.mean(), 4))

    st.markdown("""**Note:** The classification report above is based on a single 80/20 train-test split.
     The K-Fold Cross Validation gives a more stable and generalized estimate of model performance.""")


# --- SECTION: Hit/Flop Prediction ---
if section == "Hit/Flop Prediction":
    st.header("🔥 Hit or Flop Classifier")

    hit_df = spotify_df.copy()
    hit_df['is_hit'] = (hit_df['popularity'] >= 60).astype(int)

    X = hit_df.drop(columns=['track_id', 'artists', 'album_name', 'track_name', 'track_genre', 'popularity', 'is_hit'])   # dropping non-numeric/unnecessary columns and the target variable (is_hit) from the hit_df DataFrame
    y = hit_df['is_hit']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# RandomForestClassifier is an ensemble learning method that constructs multiple decision trees during training and outputs the maximum of the classes (classification) of the individual trees.
    model = RandomForestClassifier(n_estimators=100, random_state=42)   # n_estimators=100 means that the model will build 100 decision trees, and random_state=42 ensures reproducibility of the results by controlling the randomness in the model's training process.
    model.fit(X_train, y_train)

    st.subheader("🎯 Guess a Song's Hit Potential")
    song_input = st.text_input("Enter Song Name")

    if song_input:
        matches = hit_df[hit_df['track_name'].str.lower() == song_input.lower()]

        if matches.empty:
            st.error("Song not found in dataset.")
        else:
            if len(matches) > 1:
                st.write("Multiple matches found. Choose the correct artist.")
                artist_selected = st.selectbox("Select Artist", matches['artists'].unique())    
                song_row = matches[matches['artists'] == artist_selected].iloc[0]   # filters the matches DataFrame to find the row where the 'artists' column matches the artist selected by the user from the dropdown. It then selects the first row of that filtered DataFrame, which corresponds to the specific song by the chosen artist.
            else:
                song_row = matches.iloc[0]   # This line selects the first row of the matches DataFrame, which contains the details of the song that matches the user input. It is used to extract the features of that specific song for hit/flop prediction.

            song_features = song_row[X.columns].values.reshape(1, -1)    
            pred = model.predict(song_features)[0]   # the trained Random Forest model predicts whether the song is a hit (1) or a flop (0) based on its features. The predict method returns an array of predictions, and [0] retrieves the first (and only) element, giving us the predicted class for that song.
            prob = model.predict_proba(song_features)[0][1]     # the predict_proba method returns the probabilities of the song being in each class (flop or hit). [0] retrieves the first (and only) element of the resulting array, which contains the probabilities for both classes. [1] then selects the probability of the song being a hit (class 1), which is what we want to display as the hit probability.

            st.markdown(f"**{song_row['track_name']}** by *{song_row['artists']}*")
            st.success("Predicted: **Hit**" if pred == 1 else "Predicted: **Flop**")
            st.markdown(f"**Hit Probability:** `{prob:.2f}`")

    st.markdown("---")
    st.subheader("📊 Model Evaluation (Train/Test Split)")
    st.text("Classification Report on Unseen Test Data:")
    y_pred = model.predict(X_test)
    st.code(classification_report(y_test, y_pred, target_names=["Flop", "Hit"]))

    st.markdown(" **Note:** The model is evaluated on a held-out test set (20%).")


# --- SECTION 3: MOOD CLUSTERING ---
elif section == "Mood Based Clustering":
    st.header("🎭 Mood-Based Clustering")     

    selected = st.selectbox("Choose Playlist", list(playlist_files.keys()))
    st.dataframe(pd.read_csv(playlist_files[selected])[['track_name', 'artists']])   
# This line reads the CSV file corresponding to the selected playlist from the playlist_files dictionary, and then displays a DataFrame containing only the 'track_name' and 'artists' columns of that playlist. it displays the songs and their respective artists in the chosen mood-based playlist.


# --- SECTION 4: RECOMMENDER ---
elif section == "Song Recommender":
    st.header("🎯 Song Recommender")
    reco_df = spotify_df.copy()

    X_scaled = scale_features(reco_df, ['valence', 'energy', 'danceability', 'tempo', 'acousticness'])   
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)   # n_clusters=4 means that the KMeans algorithm will group the songs into 4 distinct clusters based on their audio features. random_state=42 ensures reproducibility of the clustering results, and n_init=10 specifies that the algorithm will run 10 times with different initial centroids to find the best clustering solution 
    reco_df['mood_cluster'] = kmeans.fit_predict(X_scaled)

    mood_names = {
        0: "High-Energy Electronic",
        1: "Chill Indie",
        2: "Slow Sad Acoustic",
        3: "Danceable Pop Vibes"
    }
    reco_df['playlist_name'] = reco_df['mood_cluster'].map(mood_names)   # # This line creates a new column called 'playlist_name' in the cluster_df DataFrame. It maps the numeric mood cluster labels (0, 1, 2, 3) to their corresponding descriptive playlist names using the mood_names dictionary. For example, if a song is assigned to mood_cluster 0, it will be labeled as "High-Energy Electronic" in the 'playlist_name' column.

    features = ['valence', 'energy', 'danceability', 'tempo', 'acousticness', 'instrumentalness', 'liveness', 'loudness', 'speechiness']
    reco_scaled = scale_features(reco_df, features)   # scaling the audio features listed in the features variable (valence, energy, danceability, tempo, acousticness) and stored in the reco_scaled variable as a NumPy array. This scaling is important for ensuring that all features contribute equally to the similarity calculations in the recommendation system.
    similarity_matrix = cosine_similarity(reco_scaled)   # This line computes the cosine similarity between all pairs of songs based on their scaled audio features. The resulting similarity_matrix is a square matrix where each element (i, j) represents the cosine similarity score between song i and song j. A higher score indicates that the songs are more similar in terms of their audio characteristics, which is crucial for generating accurate recommendations in the song recommender system.

# similarity_matrix looks like:    
# |   | A    | B    | C    |
# | - | ---- | ---- | ---- |
# | A | 1.00 | 0.99 | 0.15 |
# | B | 0.99 | 1.00 | 0.18 |
# | C | 0.15 | 0.18 | 1.00 |

    name = st.text_input("Enter Song Name")
    artist = st.text_input("Enter Artist Name")

    if st.button("Recommend Similar Songs"):
        idx = reco_df[(reco_df['track_name'].str.lower() == name.lower()) & 
                      (reco_df['artists'].str.lower().str.contains(artist.lower()))].index      

        if len(idx) == 0:
            st.error("Song not found. Try checking spelling.")
        else:   
            idx = idx[0]     # By taking idx[0], we are selecting the index of the first matching song, which will be used to retrieve its similarity scores from the similarity_matrix for generating recommendations
            scores = list(enumerate(similarity_matrix[idx]))    # a list of tuples called scores, where each tuple contains the index of a song and its corresponding cosine similarity score with the input song eg: [(0, 1.00), (1, 0.99), (2, 0.15)]
            scores = sorted(scores, key=lambda x: x[1], reverse=True)[1:]   # sorts the scores list in descending order based only on the similarity scores (x[1]) 
            
            shown = set()      # used to keep track of the song titles that have already been displayed as recommendations. By using a set, we can check for duplicates and ensure that each recommended song is only shown once to the user, even if multiple songs have similar features and high similarity scores.
            count = 0    
            min_similarity = 0.10    # threshold to filter out very weak recommendations

            st.subheader(f"Because you liked: {reco_df.loc[idx, 'track_name']} by {reco_df.loc[idx, 'artists']}")

            for i, score in scores:
                if score < min_similarity:   # Songs with a similarity score below this threshold will not be displayed, ensuring that only relevant and similar songs are recommended to the user.
                    continue

                song = reco_df.loc[i]
                normalized_title = normalize(song['track_name'])

                if normalized_title in shown:
                    continue

                shown.add(normalized_title)

                st.markdown(f"→ **{song['track_name']}** — {song['artists']} _({song['playlist_name']})_")

                count += 1
                if count == 5:
                    break
