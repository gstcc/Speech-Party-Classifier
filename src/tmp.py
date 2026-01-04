import pandas as pd

FILE_PATH = "../data/df_HoC_2000s.csv" 

def inspect_raw_agendas():
    print(f"Reading {FILE_PATH}...")
    
   
    df = pd.read_csv(FILE_PATH, usecols=['agenda'], nrows=10_000)
    pd.set_option('display.max_colwidth', None)
    pd.set_option('display.max_rows', 100)

    print("\n--- RANDOM SAMPLE OF 20 RAW AGENDAS ---")
    print(df['agenda'].sample(30).to_string(index=False))
    print("\n\n--- MOST COMMON AGENDAS (Top 20) ---")
    print(df['agenda'].value_counts().head(30).to_string())
    print("\n\n--- WEIRD AGENDAS (Containing '>') ---")
    nested = df[df['agenda'].str.contains('>', na=False)]
    if not nested.empty:
        print(nested['agenda'].sample(min(30, len(nested))).to_string(index=False))
    else:
        print("No nested agendas found in this sample.")

if __name__ == "__main__":
    inspect_raw_agendas()