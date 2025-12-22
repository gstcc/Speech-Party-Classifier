import pandas as pd
import re
import spacy

# Disable unused components, otherwise my computer starts to struggle
nlp = spacy.load("en_core_web_lg", disable=["parser", "ner"])
parties = ["Lab", "Con"]


def read_data(file: str):
    df = pd.read_csv(
        file,
        usecols=["speechnumber", "speaker", "party", "text"],
        index_col="speechnumber",
        nrows=1000,
    )
    return df.dropna(subset=["text", "party"])


def clean_text_basic(text: str) -> str:
    """Basic cleaning before spaCy to reduce noise."""
    if pd.isna(text):
        return ""
    text = text.lower()
    text = re.sub(r"\d+", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def preprocess(df):
    df = df[df["party"].isin(parties)].copy()
    texts = df["text"].apply(clean_text_basic).tolist()
    lemmatized_texts = []
    for doc in nlp.pipe(texts, batch_size=100):
        tokens = [
            token.lemma_ for token in doc if not token.is_stop and not token.is_punct
        ]
        lemmatized_texts.append(" ".join(tokens))

    df["tokenized"] = lemmatized_texts
    print(df[["party", "tokenized"]].head())
    return df


if __name__ == "__main__":
    PATH = "../data/df_HoC_2000s.csv"

    try:
        df = read_data(PATH)
        df_processed = preprocess(df)
        df_processed.to_csv("../data/processed_sample.csv")
    except Exception as e:
        print(f"An error occurred: {e}")
