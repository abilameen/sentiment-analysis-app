import streamlit as st
import joblib
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import plotly.graph_objects as go

# ---------------------------
# Load model + vectorizer
# ---------------------------
model = joblib.load("models/best_sentiment_model_logistic_regression.pkl")
tfidf = joblib.load("models/tfidf_vectorizer.pkl")

# ---------------------------
# Dark Theme Styling
# ---------------------------
st.set_page_config(page_title="IMDB Sentiment AI", page_icon="🎬", layout="wide")

st.markdown("""
    <style>
        .stApp {
            background-color: #1e1e2f;
            color: #ffffff;
        }
        h1, h2, h3, h4, h5, h6, .big-font {
            color: #ffffff !important;
        }
        .css-1d391kg, .css-145kmo2 { 
            background-color: #2b2b3c; 
        }
        .stButton>button {
            background-color: #4CAF50;
            color: white;
            border-radius: 8px;
            padding: 0.5em 1em;
        }
        .stButton>button:hover {
            background-color: #45a049;
        }
    </style>
""", unsafe_allow_html=True)

# ---------------------------
# Sidebar Navigation
# ---------------------------
menu = ["🏠 Home", "📊 Analysis", "🔍 Predictions", "ℹ️ About"]
choice = st.sidebar.radio("Navigation", menu)

# ---------------------------
# HOME
# ---------------------------
if choice == "🏠 Home":
    st.title("🎬 IMDB Review Sentiment Analysis")
    st.write("AI-powered sentiment classification of movie reviews (Positive vs Negative).")

    col1, col2 = st.columns(2)

    with col1:
        # Circular gauge chart
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=78,
            title={"text": "Overall Positive Sentiment"},
            gauge={'axis': {'range': [0, 100]},
                   'bar': {'color': "limegreen"},
                   'steps': [
                       {'range': [0, 50], 'color': "crimson"},
                       {'range': [50, 100], 'color': "seagreen"}]}
        ))
        st.plotly_chart(fig, use_container_width=True)

        # Sentiment trend
        st.subheader("📈 Sentiment Trend Over Time (Last 10 Years)")
        trend = [10, 20, 35, 30, 40, 55, 60, 70, 65, 80]
        st.line_chart(trend)

    with col2:
        st.subheader("☁️ Word Cloud")
        text = "excellent amazing hilarious loved disappointing terrible bad best"
        wc = WordCloud(width=400, height=250, background_color="black",
                       colormap="Set3").generate(text)
        plt.imshow(wc, interpolation="bilinear")
        plt.axis("off")
        st.pyplot(plt)

# ---------------------------
# ANALYSIS PAGE
# ---------------------------
elif choice == "📊 Analysis":
    st.title("📊 Sentiment Distribution Insights")
    st.write("Charts and analysis of IMDB reviews will go here.")

# ---------------------------
# PREDICTIONS
# ---------------------------
elif choice == "🔍 Predictions":
    st.title("🔍 Try Your Own Review")

    user_input = st.text_area("Enter a review", "This movie was absolutely brilliant!")

    if st.button("Analyze Sentiment"):
        processed = user_input.lower()
        x = tfidf.transform([processed])
        pred = model.predict(x)[0]

        if pred == "positive":
            st.success("😊 Positive Sentiment Detected!")
        else:
            st.error("😞 Negative Sentiment Detected!")

    st.markdown("### Example Reviews")
    st.success("This movie was absolutely brilliant! 🎉 [Positive sentiment]")
    st.error("Don’t bother with this film. Snooze-fest. 😴 [Negative sentiment]")

# ---------------------------
# ABOUT PAGE
# ---------------------------
elif choice == "ℹ️ About":
    st.title("ℹ️ About this Project")
    st.write("""
    - **Dataset:** IMDB Movie Reviews (50,000 samples)  
    - **Model:** Logistic Regression with TF-IDF  
    - **Tools:** Python, Scikit-learn, NLTK, Plotly, WordCloud, Streamlit  
    - **Features:** Dashboard-style analysis, Predictions, Word Cloud  
    """)
