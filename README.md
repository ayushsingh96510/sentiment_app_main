# Flask Sentiment Analyzer


A minimal Flask app that uses a scikit-learn pipeline (TF-IDF + Logistic Regression) to predict sentiment (positive/negative/neutral).


## Quick start


1. Create a virtual environment and install requirements:


```bash
python -m venv venv
source venv/bin/activate # on Windows: venv\Scripts\activate
pip install -r requirements.txt

# this is url of the kaggle data set that i have used in this project
# https://www.kaggle.com/datasets/saurabhshahane/twitter-sentiment-dataset