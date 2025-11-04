from flask import Flask, request, render_template
from model import SentimentRecommenderModel

# Initialize Flask application
app = Flask(__name__)

# Load sentiment and recommendation model
sentiment_model = SentimentRecommenderModel()


@app.route('/')
def home():
    """Render the homepage with the user input form."""
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict_recommendations():
    """
    Handle recommendation requests.
    Retrieves the username from the form, fetches product recommendations,
    and renders them back to the UI.
    """
    # Retrieve username from form input
    username = request.form.get('userName', '').lower()

    # Fetch recommendations for the user
    recommended_items = sentiment_model.getSentimentRecommendations(username)

    if recommended_items is not None:
        print(f"Fetching recommendations... Total items: {len(recommended_items)}")
        print(recommended_items)

        # RenderRecommendations table on the webpage
        return render_template(
            "index.html",
            column_names=recommended_items.columns.values,
            row_data=list(recommended_items.values.tolist()),
            zip=zip
        )
    else:
        # Display alert if user is not found
        return render_template(
            "index.html",
            message="User not found. No product recommendations available at the moment."
        )


@app.route('/predictSentiment', methods=['POST'])
def predict_sentiment():
    """
    Handle sentiment prediction for user-provided review text.
    Displays the predicted sentiment on the webpage.
    """
    # Retrieve the review text from the form
    review_text = request.form.get("reviewText", "")
    print(f"Received review text: {review_text}")

    # Predict sentiment
    predicted_sentiment = sentiment_model.classify_sentiment(review_text)
    print(f"Predicted Sentiment: {predicted_sentiment}")

    return render_template("index.html", sentiment=predicted_sentiment)


if __name__ == '__main__':
    # Run the Flask application
    app.run()
