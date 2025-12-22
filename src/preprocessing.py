import pandas as pd
import re
import spacy

# Disable unused components, otherwise my computer starts to struggle
nlp = spacy.load("en_core_web_lg", disable=["parser", "ner"])
# Using only the two biggest parties
parties = ["Lab", "Con"]
PATH = "../data/df_HoC_2000s.csv"


def read_data(file=PATH, nrows=1000):
    df = pd.read_csv(
        file,
        usecols=["speechnumber", "speaker", "party", "text"],
        index_col="speechnumber",
        nrows=nrows,
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


def balance_data(df):
    min_size = df["party"].value_counts().min()
    balanced_df = df.groupby("party").sample(n=min_size, random_state=42)
    return balanced_df


def preprocess(df, batch_size=100, sample_destination="../data/processed_sample.csv"):
    df = df[df["party"].isin(parties)].copy()
    texts = df["text"].apply(clean_text_basic).tolist()
    lemmatized_texts = []
    for doc in nlp.pipe(texts, batch_size=batch_size):
        tokens = [
            token.lemma_ for token in doc if not token.is_stop and not token.is_punct
        ]
        lemmatized_texts.append(" ".join(tokens))

    df["tokenized"] = lemmatized_texts
    # print(df[["party", "tokenized"]].head())
    df.to_csv(sample_destination)
    return df


# if __name__ == "__main__":
#
#     try:
#         df = read_data(PATH)
#         df_processed = preprocess(df)
#     except Exception as e:
#         print(f"An error occurred: {e}")
