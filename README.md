
# Automated Topic Categorization of Parliamentary Debates Using Contextual Embeddings

To run this project first this file has to be downloaded and placed into a folder called data: `https://www.kaggle.com/datasets/haotingchan/parlspeech/data?select=df_HoC_2000s.csv`

Then it can be run with 

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cd src
python models.py #If not using GPU this will take a veery long time
```