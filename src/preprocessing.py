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
    if not isinstance(agenda, str) or len(agenda) < 2:
        return "Other"

    text = agenda.lower().strip()

    noise_keywords = [
        "clause",
        "schedule",
        "orders of the day",
        "petition",
        "allotted day",
        "address",
        "business of the house",
        "point of order",
        "royal assent",
        "sittings of the house",
    ]
    if any(k in text for k in noise_keywords):
        return "Other"

    mappings = {
        "Welfare & Pensions": [
            "work and pensions",
            "social security",
            "welfare",
            "benefits",
            "disability",
            "child support",
            "poverty",
            "winter fuel",
        ],
        "Health": ["health", "nhs", "care", "hospitals", "mental", "patients"],
        "Economy": [
            "treasury",
            "budget",
            "finance",
            "economy",
            "tax",
            "business",
            "trade",
            "industry",
            "expenditure",
            "bank",
            "fina",
        ],
        "Education": [
            "education",
            "schools",
            "universities",
            "skills",
            "teaching",
            "students",
        ],
        "Democracy & Constitution": [
            "elections",
            "referendums",
            "political parties",
            "voting",
            "parliament",
            "constitutional",
            "lords",
            "house of commons",
            "electoral",
            "westminster",
        ],
        "Defense": ["defence", "armed forces", "military", "navy", "army", "raf"],
        "Foreign Policy": [
            "foreign",
            "commonwealth",
            "international",
            "european",
            "brexit",
            "diplomatic",
            "global",
            "treaty",
        ],
        "Law & Crime": [
            "home department",
            "justice",
            "police",
            "crime",
            "prison",
            "courts",
            "legal",
            "attorney",
            "solicitor",
            "law",
        ],
        "Transport": [
            "transport",
            "rail",
            "road",
            "aviation",
            "buses",
            "traffic",
            "hs2",
        ],
        "Environment": [
            "environment",
            "climate",
            "energy",
            "food",
            "rural",
            "farming",
            "water",
            "animals",
            "flood",
        ],
        "Housing & Local Govt": [
            "communities",
            "local government",
            "housing",
            "planning",
            "council",
        ],
        "Culture": ["culture", "media", "sport", "olympics", "arts", "heritage"],
        "Prime Minister (PMQs)": [
            "prime minister",
            "deputy prime minister",
            "cabinet office",
        ],
    }

    for topic, keywords in mappings.items():
        if any(k in text for k in keywords):
            return topic

    return "Other"


def clean_agenda(agenda_str, mode="auto"):
    if not isinstance(agenda_str, str):
        return ""

    clean_str = (
        agenda_str.replace("Oral Answers To Questions", "")
        .replace("Oral Answers", "")
        .strip()
    )

    if mode == "title_only":
        if "[" in clean_str:
            return clean_str.split("[")[0].strip()
        return clean_str  # If no brackets, the whole text is the title

    if mode == "breadcrumb_only":
        if "[" in clean_str and "]" in clean_str:
            try:
                content = clean_str.split("[")[1].split("]")[0]
                parts = [p.strip() for p in content.split(">")]
                # Return the last valid part that isn't truncated
                for part in reversed(parts):
                    if len(part) > 3 and not part.endswith("..."):
                        return part
            except:
                pass
        return ""  # No valid breadcrumb found

    return clean_str


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
