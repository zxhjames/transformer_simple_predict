import pandas as pd

raw = pd.read_excel('../data/LD_20142.xlsx',nrows=1000)
raw.to_excel('../data/mini.xlsx')