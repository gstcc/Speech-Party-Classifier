from preprocessing import (
    preprocess,
    read_data,
    preprocess_for_bert,
    clean_agenda,
    map_agenda_to_broad_topic,
    generate_ai_topic_map
    
)
from utils import plot_confusion_matrix, WeightedTrainer
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
from sklearn.utils.class_weight import compute_class_weight
import torch
from transformers import BertTokenizer, EarlyStoppingCallback

def get_data(nrows):
    df = read_data(nrows=nrows)

    df["crumb_raw"] = df["agenda"].apply(
        lambda x: clean_agenda(x, mode="breadcrumb_only")
    )
    df["title_raw"] = df["agenda"].apply(lambda x: clean_agenda(x, mode="title_only"))

    df["target"] = df["crumb_raw"].apply(map_agenda_to_broad_topic)

    mask_other = df["target"] == "Other"
    df.loc[mask_other, "target"] = df.loc[mask_other, "title_raw"].apply(
        map_agenda_to_broad_topic
    )

    df = df[df["target"] != "Other"]
    top_topics = df["target"].value_counts().nlargest(13).index
    df = df[df["target"].isin(top_topics)].copy()
    print("Final Topic Distribution:\n", df["target"].value_counts())
    df["tokenized"] = df["text"].str.lower().str.replace(r"[^\w\s]", "", regex=True)
    df["bert_text"] = df["text"]

    return df

def get_data_with_ai_mapping(nrows):
    
    df = read_data(nrows=nrows)
    df['clean_agenda'] = df['agenda'].apply(lambda x: clean_agenda(x, mode="breadcrumb_only"))
    mask_empty = df['clean_agenda'] == ""
    df.loc[mask_empty, 'clean_agenda'] = df.loc[mask_empty, 'agenda'].apply(lambda x: clean_agenda(x, mode="title_only"))
    unique_agendas = df['clean_agenda'].unique()
    ai_topic_map = generate_ai_topic_map(unique_agendas)
    df['target'] = df['clean_agenda'].map(ai_topic_map)
    df = df[df['target'] != "Other"]
    df = preprocess(df)
    df["bert_text"] = df["text"].apply(preprocess_for_bert)
    
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
        objective="multi:softmax",  # Set for multi-class
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
    print("\n--- RoBERTa (Optimized) ---")
    model_name = "roberta-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    unique_labels = sorted(df_train["target"].unique())
    label_map = {label: i for i, label in enumerate(unique_labels)}
    num_labels = len(unique_labels)
    
    df_train["label"] = df_train["target"].map(label_map)
    df_test["label"] = df_test["target"].map(label_map)

    def tokenize_func(examples):
        return tokenizer(examples["bert_text"], truncation=True, max_length=512)

    train_ds = Dataset.from_pandas(df_train[["bert_text", "label"]]).map(tokenize_func, batched=True)
    test_ds = Dataset.from_pandas(df_test[["bert_text", "label"]]).map(tokenize_func, batched=True)

    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(df_train["label"]),
        y=df_train["label"]
    )
    weights_tensor = torch.tensor(class_weights, dtype=torch.float)

    train_ds = train_ds.rename_column("label", "labels")
    test_ds = test_ds.rename_column("label", "labels")
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)

    
    print("Freezing bottom 6 layers of RoBERTa...")
    for name, param in model.roberta.encoder.layer.named_parameters():
        layer_num = int(name.split('.')[0])
        if layer_num < 6: 
            param.requires_grad = False

    training_args = TrainingArguments(
        output_dir="./roberta_results",
        eval_strategy="epoch",
        save_strategy="epoch",
        num_train_epochs=10,           
        learning_rate=3e-5,           
        warmup_steps=500,             
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        weight_decay=0.1,             
        label_smoothing_factor=0.1,   
        fp16=torch.cuda.is_available(),
        report_to="none",
    )

    trainer = WeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=test_ds,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        weights_tensor=weights_tensor,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
    )

    print("Training RoBERTa...")
    trainer.train()

    results = trainer.evaluate()
    print(f"\n--- BERT RESULTS ---")
    print(f"Accuracy:  {results.get('eval_accuracy', 0):.4f}")
    print(f"F1-Score:  {results.get('eval_f1', 0):.4f}")
    
    # Plotting
    print("Generating Confusion Matrix...")
    predictions_output = trainer.predict(test_ds)
    y_pred = np.argmax(predictions_output.predictions, axis=-1)
    y_true = predictions_output.label_ids
    plot_confusion_matrix(y_true, y_pred, unique_labels)


def run_experiments():
    df = get_data(nrows=50_000)
    #df = get_data_with_ai_mapping(nrows=50_000)

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
    #run_xgboost(X_train_vec, X_test_vec, y_train, y_test)

    run_bert(df_train, df_test)


if __name__ == "__main__":
    run_experiments()
