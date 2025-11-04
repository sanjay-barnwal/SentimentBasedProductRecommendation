from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import wordnet, stopwords
import pickle
import pandas as pd
import numpy as np
import re
import string
import nltk

# Download required NLTK resources
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
nltk.download('omw-1.4')


class SentimentRecommenderModel:
    """
    A combined Sentiment Classification + Product Recommendation model.
    Uses TF-IDF + XGBoost for sentiment prediction, and a collaborative filtering
    rating matrix for top-N item recommendations.
    """

    ROOT_PATH = "pickle/"
    MODEL_NAME = "sentiment-classification-xg-boost-model.pkl"
    VECTORIZER = "tfidf-vectorizer.pkl"
    RECOMMENDER = "user_final_rating.pkl"
    CLEANED_DATA = "cleaned-data.pkl"

    def __init__(self):
        """Load all required ML artifacts, vectorizers, datasets, and preprocessing tools."""
        self.model = pickle.load(open(
            self.ROOT_PATH + self.MODEL_NAME, 'rb'))

        self.vectorizer = pd.read_pickle(
            self.ROOT_PATH + self.VECTORIZER)

        self.user_final_rating = pickle.load(open(
            self.ROOT_PATH + self.RECOMMENDER, 'rb'))

        # Raw dataset for merging final product results
        self.data = pd.read_csv("dataset/sample30.csv")

        # Preprocessed text dataset for sentence-level inference
        self.cleaned_data = pickle.load(open(
            self.ROOT_PATH + self.CLEANED_DATA, 'rb'))

        # NLP tools
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))

    # -------------------------------------------------
    #   USER → TOP 20 COLLABORATIVE FILTERING PRODUCTS
    # -------------------------------------------------

    def getRecommendationByUser(self, user):
        """
        Retrieve top 20 product IDs for a user from collaborative filtering model.
        """
        return list(
            self.user_final_rating.loc[user]
            .sort_values(ascending=False)[0:20]
            .index
        )

    # -------------------------------------------------------------
    #   USER → TOP 5 SENTIMENT-ENHANCED PRODUCT RECOMMENDATIONS
    # -------------------------------------------------------------

    def getSentimentRecommendations(self, user):
        """
        Return top 5 product recommendations for a user based on:
        1. Collaborative filtering top 20 items
        2. Sentiment analysis on their reviews
        """
        if user not in self.user_final_rating.index:
            print(f"User '{user}' not found in recommendation matrix.")
            return None

        # Step 1: Get top 20 CF-based recommendations
        recommended_ids = self.getRecommendationByUser(user)

        # Filter cleaned review dataset for these product IDs
        candidate_data = self.cleaned_data[self.cleaned_data.id.isin(recommended_ids)]

        # Step 2: Convert review text → TF-IDF features
        X = self.vectorizer.transform(
            candidate_data["reviews_text_cleaned"].astype(str)
        )

        # Step 3: Predict sentiment (1 = positive)
        candidate_data["predicted_sentiment"] = self.model.predict(X)

        # Step 4: Group by product and compute % positive sentiment
        sentiment_stats = candidate_data.groupby("id", as_index=False).agg(
            total_review_count=("predicted_sentiment", "count"),
            pos_review_count=("predicted_sentiment",
                              lambda x: (x == 1).sum())
        )

        sentiment_stats["pos_sentiment_percent"] = np.round(
            (sentiment_stats["pos_review_count"] /
             sentiment_stats["total_review_count"]) * 100,
            2
        )

        # Top 5 products based on positive sentiment %
        top_products = sentiment_stats.sort_values(
            "pos_sentiment_percent", ascending=False
        ).head(5)

        # Merge metadata for UI display
        return (
            pd.merge(self.data, top_products, on="id")
            [["name", "brand", "manufacturer", "pos_sentiment_percent"]]
            .drop_duplicates()
            .sort_values(["pos_sentiment_percent", "name"], ascending=[False, True])
        )

    # -------------------------------------------------
    #   SENTIMENT CLASSIFICATION FOR ANY REVIEW TEXT
    # -------------------------------------------------

    def classify_sentiment(self, review_text):
        """
        Predict the sentiment (1 = positive, 0 = negative)
        for a single user review.
        """
        cleaned_text = self.preprocess_text(review_text)
        X = self.vectorizer.transform([cleaned_text])
        return self.model.predict(X)

    # -------------------------------------------------
    #   TEXT CLEANING AND NLP PROCESSING
    # -------------------------------------------------

    def preprocess_text(self, text):
        """
        Clean text by:
        - lowercasing
        - removing punctuation, digits
        - removing stopwords
        - lemmatizing words using POS tags
        """
        text = text.lower().strip()
        text = re.sub(r"\[\s*\w*\s*\]", "", text)
        text = text.translate(str.maketrans('', '', string.punctuation))
        text = re.sub(r"\S*\d\S*", "", text)

        return self.lemma_text(text)

    def get_wordnet_pos(self, tag):
        """Map POS tag to WordNet format."""
        if tag.startswith('J'):
            return wordnet.ADJ
        elif tag.startswith('V'):
            return wordnet.VERB
        elif tag.startswith('N'):
            return wordnet.NOUN
        elif tag.startswith('R'):
            return wordnet.ADV
        return wordnet.NOUN

    def remove_stopword(self, text):
        """
        Remove non-alphabetic tokens and stopwords.
        """
        words = [
            word for word in text.split()
            if word.isalpha() and word not in self.stop_words
        ]
        return " ".join(words)

    def lemma_text(self, text):
        """
        Apply POS tagging and lemmatization on filtered words.
        """
        tokens = word_tokenize(self.remove_stopword(text))
        pos_tags = nltk.pos_tag(tokens)

        lemmatized_words = [
            self.lemmatizer.lemmatize(word, self.get_wordnet_pos(pos))
            for word, pos in pos_tags
        ]

        return " ".join(lemmatized_words)
