import pandas as pd
import re
import spacy
import random

# Disable unused components, otherwise my computer starts to struggle
nlp = spacy.load("en_core_web_lg", disable=["parser", "ner"])
# Using only the two biggest parties
parties = ["Lab", "Con"]
PATH = "../data/df_HoC_2000s.csv"

def map_agenda_to_broad_topic(agenda):
    if not isinstance(agenda, str):
        return None
    text = agenda.strip()
    return agenda

def clean_agenda(agenda_str, mode="title"):
    if not isinstance(agenda_str, str):
        return ""  # Return empty string instead of None to prevent errors


    clean_str = (
        agenda_str.replace("Oral Answers To Questions", "")
        .replace("Oral Answers", "")
        .strip()
        .lower()
    )
    # First try to get e.g Social Security from > Social Security] part, otherwise take first sentence
    if "[" in clean_str and "]" in clean_str:
        try:
            content = clean_str.split("[")[1].split("]")[0]
            parts = [p.strip() for p in content.split(">")]
            for part in reversed(parts):
                if len(part) > 3 and not part.endswith("..."):
                    return part
        except:
            pass
    elif "[" in clean_str:
        return clean_str.split("[")[0].strip()


    return clean_str  # Fallback


def read_data(file=PATH, nrows=1000):
    approx_total = 300_000
    p = (nrows * 1.5) / approx_total

    def keep_row(index):
        if index == 0:
            return False
        return random.random() > p

    df = pd.read_csv(
        file,
        usecols=["speechnumber", "speaker", "party", "text", "agenda"],
        index_col="speechnumber",
        skiprows=keep_row,
    )

    df = df.dropna(subset=["text", "party"])
    if len(df) > nrows:
        df = df.sample(n=nrows)

    return df


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


def preprocess_for_bert(text: str) -> str:
    if pd.isna(text):
        return ""
    # Only remove extra whitespace/newlines
    text = re.sub(r"\s+", " ", text)
    return text.strip()


# if __name__ == "__main__":
#
#     try:
#         df = read_data(PATH)
#         df_processed = preprocess(df)
#     except Exception as e:
#         print(f"An error occurred: {e}")
