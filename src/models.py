from preprocessing import preprocess, read_data, balance_data, preprocess_for_bert
from utils import plot_confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    precision_recall_fscore_support,
)
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)


def get_data(nrows):
    df = read_data(nrows=nrows)
    df = preprocess(df)  # This creates df['tokenized']
    df["bert_text"] = df["text"].apply(preprocess_for_bert)
    df = balance_data(df)
    return df


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
        tree_method="hist",
    )

    print("Training XGBoost...")
    model.fit(X_train_vec, y_train_num)

    # 3. Predict
    preds_num = model.predict(X_test_vec)
    preds = le.inverse_transform(preds_num)

    print("\n--- XGBOOST RESULTS ---")
    print(classification_report(y_test, preds))
    return model


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)

    # Calculate metrics
    acc = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average="binary"
    )

    # These keys MUST match your print statements later
    return {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall}


def run_bert(df_train, df_test):
    model_name = "distilbert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # 1. Prepare Labels (0 = Con, 1 = Lab)
    label_map = {"Con": 0, "Lab": 1}
    df_train["label"] = df_train["party"].map(label_map)
    df_test["label"] = df_test["party"].map(label_map)

    # 2. Tokenization Function
    def tokenize_func(examples):
        return tokenizer(
            examples["bert_text"], padding="max_length", truncation=True, max_length=256
        )

    # 3. Convert to HF Datasets
    train_ds = Dataset.from_pandas(df_train[["bert_text", "label"]]).map(
        tokenize_func, batched=True
    )
    test_ds = Dataset.from_pandas(df_test[["bert_text", "label"]]).map(
        tokenize_func, batched=True
    )

    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

    training_args = TrainingArguments(
        output_dir="./bert_results",
        eval_strategy="epoch",
        save_strategy="epoch",
        num_train_epochs=4,
        learning_rate=2e-5,
        warmup_steps=100,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        fp16=False,  # False=Cpu, True=GPU
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=test_ds,
        compute_metrics=compute_metrics,
    )

    print("Training BERT (this may take a while)...")
    trainer.train()

    # 5. Evaluate
    results = trainer.evaluate()
    print(f"\n--- BERT RESULTS ---")
    print(f"Accuracy:  {results.get('eval_accuracy', 0):.4f}")
    print(f"F1-Score:  {results.get('eval_f1', 0):.4f}")

    print("Generating Confusion Matrix...")
    predictions_output = trainer.predict(test_ds)
    y_pred = np.argmax(predictions_output.predictions, axis=-1)
    y_true = predictions_output.label_ids
    plot_confusion_matrix(y_true, y_pred)


def run_experiments():
    df = get_data(nrows=20_000)

    df_train, df_test = train_test_split(df, test_size=0.2, random_state=42)

    X_train_text = df_train["tokenized"]
    X_test_text = df_test["tokenized"]
    y_train = df_train["party"]
    y_test = df_test["party"]

    vectorizer = TfidfVectorizer(
        stop_words="english", max_features=10000, ngram_range=(1, 2)
    )
    X_train_vec = vectorizer.fit_transform(X_train_text)
    X_test_vec = vectorizer.transform(X_test_text)  # print("--- NAIVE BAYES ---")

    # run_naive_bayes(X_train_vec, X_test_vec, y_train, y_test)
    #
    # run_logistic_regression(X_train_vec, X_test_vec, y_train, y_test)
    #
    # run_xgboost(X_train_vec, X_test_vec, y_train, y_test)

    print("\n--- BERT ---")
    run_bert(df_train, df_test)


if __name__ == "__main__":
    run_experiments()
