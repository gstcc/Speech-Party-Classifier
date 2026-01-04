from preprocessing import (
    preprocess,
    read_data,
    preprocess_for_bert,
    clean_agenda,
    map_agenda_to_broad_topic,
)
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
    DataCollatorWithPadding,
)


def get_data(nrows):
    df = read_data()

    df["title_raw"] = df["agenda"].apply(lambda x: clean_agenda(x, mode="title_only"))
    df["crumb_raw"] = df["agenda"].apply(
        lambda x: clean_agenda(x, mode="breadcrumb_only")
    )

    df["target"] = df["title_raw"].apply(map_agenda_to_broad_topic)

    mask_other = df["target"] == "Other"
    df.loc[mask_other, "target"] = df.loc[mask_other, "crumb_raw"].apply(
        map_agenda_to_broad_topic
    )

    df = df[df["target"] != "Other"]
    top_topics = df["target"].value_counts().nlargest(13).index
    df = df[df["target"].isin(top_topics)].copy()
    print("Final Topic Distribution:\n", df["target"].value_counts())
    df["tokenized"] = df["text"].str.lower().str.replace(r"[^\w\s]", "", regex=True)
    df["bert_text"] = df["text"]

    return df


def run_naive_bayes(X_train, X_test, y_train, y_test):
    print("\n--- NAIVE BAYES ---")
    nb = MultinomialNB()
    nb.fit(X_train, y_train)
    preds = nb.predict(X_test)
    print(classification_report(y_test, preds))


def run_logistic_regression(X_train, X_test, y_train, y_test):
    print("\n--- LOGISTIC REGRESSION ---")
    model = LogisticRegression(
        class_weight="balanced", max_iter=2000, multi_class="auto"
    )
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    print(classification_report(y_test, preds))
    return model


def run_xgboost(X_train_vec, X_test_vec, y_train, y_test):
    print("\n--- XGBOOST ---")
    le = LabelEncoder()
    y_train_num = le.fit_transform(y_train)
    y_test_num = le.transform(y_test)
    model = XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        tree_method="hist",
        objective="multi:softmax",  # Explicitly set for multi-class
        num_class=len(le.classes_),
    )

    print("Training XGBoost...")
    model.fit(X_train_vec, y_train_num)

    # 3. Predict
    preds_num = model.predict(X_test_vec)
    preds = le.inverse_transform(preds_num)

    print(classification_report(y_test, preds))
    return model


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average="weighted", zero_division=0
    )

    return {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall}


def run_bert(df_train, df_test):
    print("\n--- BERT ---")
    model_name = "distilbert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    unique_labels = sorted(df_train["target"].unique())
    label_map = {label: i for i, label in enumerate(unique_labels)}
    num_labels = len(unique_labels)
    print(f"BERT training on {num_labels} topics.")
    df_train["label"] = df_train["target"].map(label_map)
    df_test["label"] = df_test["target"].map(label_map)

    def tokenize_func(examples):
        return tokenizer(examples["bert_text"], truncation=True, max_length=256)

    train_ds = Dataset.from_pandas(df_train[["bert_text", "label"]]).map(
        tokenize_func, batched=True
    )
    test_ds = Dataset.from_pandas(df_test[["bert_text", "label"]]).map(
        tokenize_func, batched=True
    )

    train_ds = train_ds.rename_column("label", "labels")
    test_ds = test_ds.rename_column("label", "labels")

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=num_labels
    )

    training_args = TrainingArguments(
        output_dir="./bert_agenda_results",
        eval_strategy="epoch",
        save_strategy="epoch",
        num_train_epochs=3,
        learning_rate=2e-5,
        warmup_steps=100,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        fp16=True,  # Set True if you have GPU
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=test_ds,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    print("Training BERT (this may take a while)...")
    trainer.train()

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
    # Load data (Agenda Task)
    df = get_data(nrows=20_000)

    # Use 'target' (the cleaned agenda) as y
    df_train, df_test = train_test_split(
        df, test_size=0.2, random_state=42, stratify=df["target"]
    )

    X_train_text = df_train["tokenized"]
    X_test_text = df_test["tokenized"]
    y_train = df_train["target"]
    y_test = df_test["target"]

    vectorizer = TfidfVectorizer(
        stop_words="english", max_features=10000, ngram_range=(1, 2)
    )
    X_train_vec = vectorizer.fit_transform(X_train_text)
    X_test_vec = vectorizer.transform(X_test_text)

    run_naive_bayes(X_train_vec, X_test_vec, y_train, y_test)
    run_logistic_regression(X_train_vec, X_test_vec, y_train, y_test)
    run_xgboost(X_train_vec, X_test_vec, y_train, y_test)

    run_bert(df_train, df_test)


if __name__ == "__main__":
    run_experiments()
