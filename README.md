# Sentiment-Based Product Recommendation System

## 📌 Problem Statement

The landscape of modern commerce has transformed significantly, with e-commerce becoming a dominant force. Traditional brick-and-mortar models have given way to digital platforms that allow businesses to reach customers directly. Market leaders such as Amazon and Flipkart have set high standards by offering vast product selections with seamless accessibility.

In this competitive space, **Ebuss** aims to strengthen its presence across various product categories, ranging from household essentials to electronics. To stay ahead, the company must innovate and leverage technology to enhance customer experience.

As a Machine Learning Engineer at Ebuss, the objective is to develop a **sentiment-driven product recommendation system** that improves recommendation accuracy by utilizing customer feedback.

This project focuses on:

1. **Data Acquisition & Sentiment Analysis** – Collect and analyze user reviews to extract sentiment.
2. **Recommendation Engine Development** – Build a robust recommendation system using sentiment insights.
3. **Sentiment-Integrated Enhancements** – Refine recommendations using predicted sentiment scores.
4. **End-to-End Deployment** – Deliver a smooth and intuitive user interface for seamless system interaction.

Understanding customer preferences is crucial in today’s fast-evolving e-commerce landscape. A sentiment-enhanced recommendation system allows Ebuss to offer more personalized and satisfying product suggestions, ultimately enriching customer experience.

---

## ✅ Solution

GitHub Repository:  
**https://github.com/sanjay-barnwal/SentimentBasedProductRecommendation**

---

## 🛠️ Built With

- **Python** 3.9.12  
- **scikit-learn** 1.4.1.post1  
- **xgboost** 2.0.3  
- **numpy** 1.26.4  
- **nltk** 3.8.1  
- **pandas** 2.2.1  
- **Flask** 3.0.2  
- **Bootstrap CDN** 5.1.3  

---

## 🚀 Solution Approach

- The dataset and attribute descriptions are provided in the `dataset/` folder.
- Initial steps involve **data cleaning**, **EDA**, and **text preprocessing** using NLP techniques.  
  TF-IDF vectorization is applied to convert combined review text into numerical form.
- To resolve **class imbalance**, the **SMOTE oversampling technique** is used before model training.
- Multiple ML models are developed, including:
  - Logistic Regression  
  - Naive Bayes  
  - Decision Tree  
  - Random Forest  
  - XGBoost  

  Models are evaluated using **Accuracy**, **Precision**, **Recall**, **F1-Score**, and **AUC**, with **XGBoost emerging as the best performer**.

- A **Collaborative Filtering Recommender System** is implemented using both **User-User** and **Item-Item** methods, with RMSE used for evaluation.  
  Detailed implementation is documented in `SentimentBasedProductRecommendation.ipynb`.

- For **product sentiment prediction**, the recommender system identifies the top 20 products.  
  Sentiment is predicted for all reviews, and the **top 5 products with the highest positive sentiment** are selected (implemented in `model.py`).

- Machine Learning models are saved as **pickle files** in the `pickle/` directory.

- A **Flask API (`app.py`)** is developed for model inference, and a simple user interface is created using **Bootstrap and Jinja templates** (`templates/index.html`).

---

