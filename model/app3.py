import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import joblib
import os
import re

# =========================
# Text Cleaning Function
# =========================
url_re = re.compile(r"https?://\S+|www\.\S+")
user_re = re.compile(r"@\w+")
hash_re = re.compile(r"#")
space_re = re.compile(r"\s+")

def clean_text(s):
    s = str(s).lower()
    s = url_re.sub(" ", s)
    s = user_re.sub(" ", s)
    s = hash_re.sub("", s)
    s = space_re.sub(" ", s).strip()
    return s

# =========================
# Load the trained model safely
# =========================
def load_model():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_dir, "sam1.pkl")

    if not os.path.exists(model_path):
        model_path = os.path.join(os.path.dirname(current_dir), "sam1.pkl")

    if not os.path.exists(model_path):
        st.error(f"❌ Model file not found. Expected at: {model_path}")
        raise FileNotFoundError("sam1.pkl not found")

    # Unpack the tuple (TF-IDF vectorizer, trained model)
    tfidf, model = joblib.load(model_path)
    return tfidf, model

tfidf, pipe_lr = load_model()

# =========================
# Emojis for each emotion
# =========================
emotions_emoji_dict = {
    "anger": "😠",
    "disgust": "🤮",
    "fear": "😨😱",
    "happy": "🤗",
    "joy": "😂",
    "neutral": "😐",
    "sad": "😔",
    "sadness": "😔",
    "shame": "😳",
    "surprise": "😮"
}

# =========================
# Prediction Helpers
# =========================
def predict_emotions(docx):
    docx_clean = clean_text(docx)
    vect = tfidf.transform([docx_clean])
    results = pipe_lr.predict(vect)
    return results[0]

def get_prediction_proba(docx):
    docx_clean = clean_text(docx)
    vect = tfidf.transform([docx_clean])
    if hasattr(pipe_lr, "predict_proba"):
        return pipe_lr.predict_proba(vect)
    else:
        # LinearSVC does not support predict_proba; create dummy one-hot probabilities
        pred_class = pipe_lr.predict(vect)[0]
        proba = [1.0 if cls == pred_class else 0.0 for cls in pipe_lr.classes_]
        return np.array([proba])

# =========================
# Streamlit App
# =========================
def main():
    st.title("✨😊 Text Emotion Detection")
    st.subheader("Enter text and detect its emotion instantly!")

    with st.form(key='my_form'):
        raw_text = st.text_area("Type Here")
        submit_text = st.form_submit_button(label='Submit')

    if submit_text and raw_text.strip() != "":
        col1, col2 = st.columns(2)

        prediction = predict_emotions(raw_text)
        probability = get_prediction_proba(raw_text)

        with col1:
            st.success("Original Text")
            st.write(raw_text)

            st.success("Prediction")
            emoji_icon = emotions_emoji_dict.get(prediction, "❓")
            st.markdown(f"<h1 style='font-size:80px;'>{emoji_icon}</h1>", unsafe_allow_html=True)
            st.subheader(f"Prediction: {prediction}")

            st.write(f"Confidence: {np.max(probability):.2f}")

        with col2:
            st.success("Prediction Probability")
            proba_df = pd.DataFrame(probability, columns=pipe_lr.classes_)

            proba_df_clean = proba_df.T.reset_index()
            proba_df_clean.columns = ["emotions", "probability"]

            fig = alt.Chart(proba_df_clean).mark_bar().encode(
                x='emotions',
                y='probability',
                color='emotions'
            )
            st.altair_chart(fig, use_container_width=True)

if __name__ == '__main__':
    main()
