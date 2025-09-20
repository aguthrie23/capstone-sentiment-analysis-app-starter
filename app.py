from flask import Flask, render_template, request
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
app = Flask(__name__)
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        user_text = request.form.get("user_text")
        sid = SentimentIntensityAnalyzer()
        sentiment = sid.polarity_scores(user_text)
        return render_template('form.html', sentiment=sentiment)
    return render_template('form.html')
if __name__ == "__main__":
    app.run()
