import pandas as pd
import re
import spacy
import random
from transformers import pipeline
from tqdm import tqdm
import pandas as pd
import torch
import os
import json

# Disable unused components, otherwise my computer starts to struggle
nlp = spacy.load("en_core_web_lg", disable=["parser", "ner"])
# Using only the two biggest parties
parties = ["Lab", "Con"]
PATH = "../data/df_HoC_2000s.csv"


def generate_ai_topic_map(unique_agendas, cache_file="ai_topic_cache.json"):
    candidate_labels = [
        "Health", "Education", "Transport", "Welfare & Pensions", 
        "Environment", "Law & Crime", "Housing & Local Govt", 
        "Culture", "International Affairs", "Democracy & Constitution", "Economy"
    ]
    
    classifier = pipeline("zero-shot-classification", model="valhalla/distilbart-mnli-12-1", device=(0 if torch.cuda.is_available() else -1)) # Use GPU if available
 

    mapping = {}
    

    print(f"AI is classifying {len(unique_agendas)} unique topics in batches...")
    results = classifier(list(unique_agendas), candidate_labels, batch_size=16)

    for agenda, res in zip(unique_agendas, results):
        if res['scores'][0] > 0.4:
            mapping[agenda] = res['labels'][0]
        else:
            mapping[agenda] = "Other"
        
    return mapping

def map_agenda_to_broad_topic(agenda):
    if not isinstance(agenda, str):
        return None
    
    text = agenda.lower().strip()
    
    # 1. Noise Filter (Expanded based on your data)
    noise_keywords = [
        'points of order', 'business of the house', 'sittings of the house', 
        'royal assent', 'speaker\'s statement', 'petition'
    ]
    if any(k in text for k in noise_keywords):
        return None 

    if any(k in text for k in ['health', 'nhs', 'care', 'hospitals', 'mental', 'patients', 'doctors', 'nurses', 'medical', 'euthanasia']):
        return 'Health'
        
    if any(k in text for k in ['education', 'schools', 'universities', 'skills', 'teaching', 'students']):
        return 'Education'
        
    if any(k in text for k in ['transport', 'rail', 'road', 'aviation', 'buses', 'traffic', 'vehicle emissions', 'drivers']):
        return 'Transport'
        
    if any(k in text for k in ['social security', 'welfare', 'benefits', 'disability', 'child support', 'poverty', 'pensions', 'families tax credit']):
        return 'Welfare & Pensions'
        
    if any(k in text for k in ['environment', 'climate', 'energy', 'food', 'rural', 'farming', 'agriculture', 'fisheries', 'water', 'animals', 'bovine', 'badgers', 'gmos', 'biotechnology', 'acid rain']):
        return 'Environment'
        
    if any(k in text for k in ['home department', 'justice', 'police', 'crime', 'prison', 'courts', 'legal', 'magistrates', 'sexual offences', 'cryptography']):
        return 'Law & Crime'
        
    if any(k in text for k in ['communities', 'local government', 'housing', 'planning', 'council']):
        return 'Housing & Local Govt'
        
    if any(k in text for k in ['culture', 'media', 'sport', 'olympics', 'arts']):
        return 'Culture'

    if any(k in text for k in ['defence', 'armed forces', 'military', 'navy', 'army', 'foreign', 'commonwealth', 'international', 'european', 'yugoslavia', 'asylum', 'immigration']):
        return 'International Affairs'

    # General stuff, added last to avoid confusion for model, since basically everything is about economy especially
    if any(k in text for k in ['elections', 'referendums', 'political parties', 'voting', 'parliament', 'constitutional', 'representation of the people', 'electoral', 'disqualifications']):
        return 'Democracy & Constitution'

    if any(k in text for k in ['treasury', 'budget', 'finance', 'economy', 'tax', 'business', 'trade', 'industry', 'financial services', 'fair trading', 'utilities', 'competitiveness', 'debt']):
        return 'Economy'
            
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
        return clean_str

    if mode == 'breadcrumb_only':
        if '[' in clean_str and ']' in clean_str:
            try:
                content = clean_str.split('[')[-1].split(']')[0]
                parts = [p.strip() for p in content.split('>')]
                
                
                for part in reversed(parts):
                    lower_part = part.lower()
                    # Skip garbage, sectioon 1 as label says nothing about actual topic
                    if any(x in lower_part for x in ["clause", "schedule", "orders of the day", "programme", "resolution"]):
                        continue
                    if len(part) < 3 or part.endswith("..."):
                        continue
                    
                    return part 
                
                return clean_str.split("[")[0].strip()
            except:
                pass
        return "" 

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
