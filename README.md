# 🎬 Sentiment Analysis of IMDB Movie Reviews  

## 📌 Project Overview  
This project is an **AI-powered Sentiment Analysis App** that classifies IMDB movie reviews as **Positive 😊** or **Negative 😞**.  
It was built using **Natural Language Processing (NLP)** and **Machine Learning** with a clean, professional **Streamlit web interface**.  

---

## 🛠 Features  
- 📊 **Dashboard-style UI** with dark theme  
- 🔍 **Single Review Prediction** (with confidence score & probability breakdown)  
- 🧪 **Batch Testing** with multiple sample reviews  
- ☁️ **Word Cloud** visualization for most common words  
- 📈 **Sentiment Trend Analysis** chart  
- ✅ **Interactive Web App** built using **Streamlit**  

---

## 📂 Project Structure  
```
sentiment-analysis-project/
├── data/
│   ├── raw/               # Raw IMDB dataset
│   └── processed/         # Cleaned & preprocessed data
├── models/                # Trained ML models + TF-IDF vectorizer
├── notebooks/             # Jupyter notebooks (exploration, preprocessing, training, demo)
├── app.py                 # Streamlit web app
├── requirements.txt       # Dependencies
└── README.md
```

---

## ⚙️ Installation & Setup  
1. Clone this repository:  
   ```bash
   git clone <repo-url>
   cd sentiment-analysis-project
   ```
2. Create and activate a virtual environment:  
   ```bash
   python -m venv sentiment_env
   source sentiment_env/bin/activate   # Mac/Linux
   sentiment_env\Scripts\activate      # Windows
   ```
3. Install dependencies:  
   ```bash
   pip install -r requirements.txt
   ```
4. Download the IMDB dataset (`IMDB_Dataset.csv`) from Kaggle and place it inside `data/raw/`.  

---

## ▶️ Running the App  
1. Train the model (via notebooks 01–03).  
2. Launch the Streamlit app:  
   ```bash
   streamlit run app.py
   ```
3. Open your browser → [http://localhost:8501](http://localhost:8501)  

---

## 📊 Technologies Used  
- **Python 3.8+**  
- **Jupyter Notebook**  
- **Libraries:** pandas, numpy, scikit-learn, nltk, matplotlib, seaborn, wordcloud, plotly, joblib  
- **Streamlit** (for web app UI)  

---

## 🚀 Future Enhancements  
- Add **neutral sentiment** (multi-class classification).  
- Deploy on **Streamlit Cloud / Heroku / Render**.  
- Extend to **Twitter / Product Reviews** datasets.  
- Try advanced models like **LSTM** or **BERT**.  
