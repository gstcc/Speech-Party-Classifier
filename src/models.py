from preprocessing import preprocess, read_data, balance_data
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
import pandas as pd


def get_data(nrows):
    df = read_data(nrows=nrows)
    df = preprocess(df)
    df = balance_data(df)
    return df


def run_experiments():
    df = get_data(nrows=10_000)

    X_train, X_test, y_train, y_test = train_test_split(
        df["tokenized"], df["party"], test_size=0.2, random_state=42
    )

    vectorizer = TfidfVectorizer(
        stop_words="english", max_features=10000, ngram_range=(1, 2)
    )
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    print("--- NAIVE BAYES ---")
    run_naive_bayes(X_train_vec, X_test_vec, y_train, y_test)

    print("\n--- LOGISTIC REGRESSION ---")
    lr_model = run_logistic_regression(X_train_vec, X_test_vec, y_train, y_test)
    run_xgboost(X_train_vec, X_test_vec, y_train, y_test)

    # Optional: Extract the partisan words here!
    # get_top_partisan_words(lr_model, vectorizer)


def run_naive_bayes(X_train, X_test, y_train, y_test):
    nb = MultinomialNB()
    nb.fit(X_train, y_train)
    preds = nb.predict(X_test)
    print(classification_report(y_test, preds))


def run_logistic_regression(X_train, X_test, y_train, y_test):
    model = LogisticRegression(class_weight="balanced", max_iter=1000)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    print(classification_report(y_test, preds))
    return model


def run_xgboost(X_train_vec, X_test_vec, y_train, y_test):
    # 1. Encode labels
    le = LabelEncoder()
    y_train_num = le.fit_transform(y_train)
    y_test_num = le.transform(y_test)

    # 2. Initialize model
    model = XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        # This allows XGBoost to handle the Sparse Matrix directly
        tree_method="hist",
    )

    print("Training XGBoost...")
    # X_train_vec is already numerical, so this will now work
    model.fit(X_train_vec, y_train_num)

    # 3. Predict
    preds_num = model.predict(X_test_vec)
    preds = le.inverse_transform(preds_num)

    print("\n--- XGBOOST RESULTS ---")
    print(classification_report(y_test, preds))
    return model


def bert():
    pass


run_experiments()
