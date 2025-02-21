import streamlit as st
import pandas as pd
import joblib
import spacy
import nltk
import re
import time
import matplotlib.pyplot as plt
import seaborn as sns
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from webdriver_manager.chrome import ChromeDriverManager
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim import corpora, models

# ✅ Download necessary resources
nltk.download("vader_lexicon")

# ✅ Load NLP models
nlp = spacy.load("en_core_web_sm")  # NER Model
sia = SentimentIntensityAnalyzer()  # Sentiment Analysis Model

# 🎬 **Streamlit UI**
st.title("📊 YouTube Audience Engagement Analysis")
video_url = st.text_input("🔗 Enter YouTube Video URL:")

if st.button("Analyze"):
    if "youtube.com/watch" in video_url:
        st.info("📥 Fetching comments... Please wait.")

        # ✅ **Setup Chrome WebDriver (Headless)**
        chrome_options = Options()
        chrome_options.add_argument("--headless=new")  # Ensures fully headless mode
        chrome_options.add_argument("--disable-gpu")
        chrome_options.add_argument("--window-size=1920x1080")
        chrome_options.add_argument("--disable-extensions")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")

        try:
            service = Service(ChromeDriverManager().install())
            driver = webdriver.Chrome(service=service, options=chrome_options)

            # Open YouTube video
            driver.get(video_url)
            time.sleep(5)

            # Scroll to load comments
            for _ in range(50):
                driver.find_element(By.TAG_NAME, 'body').send_keys(Keys.END)
                time.sleep(2)

            # Extract comments
            comments = driver.find_elements(By.XPATH, '//*[@id="content-text"]')
            extracted_comments = [comment.text.strip() for comment in comments]

            # Close Selenium
            driver.quit()

        except Exception as e:
            st.error(f"❌ Selenium Error: {e}")
            driver.quit()
            extracted_comments = []

        if not extracted_comments:
            st.error("❌ No comments found! Try another video.")
        else:
            # ✅ **Function to clean comments**
            def clean_text(text):
                text = text.lower().strip()
                text = re.sub(r"http\S+|www\S+", "", text)  # Remove URLs
                text = re.sub(r"[^a-zA-Z0-9\s]", "", text)  # Remove special characters
                return text

            # Apply cleaning
            cleaned_comments = [clean_text(comment) for comment in extracted_comments]

            # ✅ **Function to get sentiment**
            def get_sentiment(text):
                sentiment_score = sia.polarity_scores(text)["compound"]
                if sentiment_score >= 0.05:
                    return "Positive"
                elif sentiment_score <= -0.05:
                    return "Negative"
                else:
                    return "Neutral"

            # Apply sentiment analysis
            sentiments = [get_sentiment(comment) for comment in cleaned_comments]

            # ✅ **Function to extract named entities**
            def extract_entities(text):
                doc = nlp(text)
                entities = [(ent.text, ent.label_) for ent in doc.ents]
                return entities if entities else None

            # Apply NER
            entity_results = [extract_entities(comment) for comment in extracted_comments]

            # 🎯 **TF-IDF Keyword Extraction**
            vectorizer = TfidfVectorizer(stop_words="english", max_features=10)
            tfidf_matrix = vectorizer.fit_transform(cleaned_comments)
            keywords = vectorizer.get_feature_names_out()

            # 🎯 **LDA Topic Modeling**
            tokenized_comments = [comment.split() for comment in cleaned_comments]
            dictionary = corpora.Dictionary(tokenized_comments)
            corpus = [dictionary.doc2bow(comment) for comment in tokenized_comments]
            lda_model = models.LdaModel(corpus, num_topics=3, id2word=dictionary, passes=10)
            topics = lda_model.print_topics(num_words=5)

            # ✅ **Create DataFrame**
            results_df = pd.DataFrame({
                "Comment": extracted_comments,
                "Sentiment": sentiments,
                "Entities": entity_results
            })

            # ✅ **Sentiment Distribution**
            sentiment_counts = results_df["Sentiment"].value_counts()

            # 📊 **1️⃣ Sentiment Analysis - Bar Chart**
            st.write("### 📊 Sentiment Analysis of YouTube Comments")
            fig, ax = plt.subplots(figsize=(8, 5))
            sns.barplot(x=sentiment_counts.index, y=sentiment_counts.values, palette=["green", "blue", "red"], ax=ax)
            plt.xlabel("Sentiment")
            plt.ylabel("Count")
            plt.title("Sentiment Distribution")
            st.pyplot(fig)

            # 📊 **2️⃣ Sentiment Analysis - Pie Chart**
            st.write("### 📊 Sentiment Distribution")
            fig, ax = plt.subplots(figsize=(6, 6))
            colors = ["green", "blue", "red"]
            plt.pie(sentiment_counts, labels=sentiment_counts.index, autopct="%1.1f%%", colors=colors, startangle=140)
            plt.title("Sentiment Breakdown")
            st.pyplot(fig)

            # 📊 **3️⃣ Named Entity Recognition - Bar Chart**
            entity_counts = {}
            for entities in entity_results:
                if entities:
                    for entity in entities:
                        entity_counts[entity[1]] = entity_counts.get(entity[1], 0) + 1

            if entity_counts:
                st.write("### 📊 Named Entity Distribution")
                fig, ax = plt.subplots(figsize=(8, 5))
                sns.barplot(x=list(entity_counts.keys()), y=list(entity_counts.values()), palette="coolwarm", ax=ax)
                plt.xticks(rotation=45)
                st.pyplot(fig)
            else:
                st.warning("⚠ No significant named entities found.")

            # 📊 **4️⃣ Top 10 Keywords (TF-IDF)**
            st.write("### 🔑 Top Keywords from Comments")
            st.write(", ".join(keywords))

            # 📊 **5️⃣ Top Topics (LDA)**
            st.write("### 🔥 Key Topics Discussed")
            for topic in topics:
                st.write(f"🔹 {topic}")

            # ✅ **Display sample comments**
            st.write("### 💬 Sample Comments with Sentiments & Entities")
            st.write(results_df.head(20))

    else:
        st.error("❌ Invalid YouTube URL! Please enter a valid video link.")
