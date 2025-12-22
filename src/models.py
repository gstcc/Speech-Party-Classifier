from preprocessing import preprocess, read_data, balance_data
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd


def naive_bayers():
    df = read_data(nrows=20_000)
    df = preprocess(df)
    df = balance_data(df)
    print(df.head())
    X_train, X_test, y_train, y_test = train_test_split(
        df["tokenized"], df["party"], test_size=0.2, random_state=42
    )

    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_features=10000,
        ngram_range=(
            1,
            2,
        ),
    )
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    nb = MultinomialNB()
    nb.fit(X_train_vec, y_train)

    preds = nb.predict(X_test_vec)
    print(classification_report(y_test, preds))


def linear_regression():
    pass


def xg_boost():
    pass


def bert():
    pass


naive_bayers()
