import pandas as pd
import cleaner.TextCleaner as tc
from way import pickle1, rcsv


df = pd.read_csv(rcsv)
print('Подготовка данных...')
df = tc.CleanText.prepare_df(df, pickle=pickle1)
