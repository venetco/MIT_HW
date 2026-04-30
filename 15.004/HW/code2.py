# Imports
import numpy as np
import pandas as pd
from google.colab import files

uploaded = files.upload()
csvFileName = next(iter(uploaded))

def calcCorrelations (csvFileName):
  pricesDF = pd.read_csv (csvFileName, index_col = 0)

  # clean invalid data : set to missing
  pricesDF[pricesDF <= 0] = np.nan

  returnsDF = pricesDF.pct_change(1,fill_method = None)
  correlations = returnsDF.corr()
  return correlations

print(calcCorrelations(csvFileName))


# Question 3
print(calcCorrelations(csvFileName).loc['AAPL']['GOOGL'])

# Question 4
print(calcCorrelations(csvFileName).loc['AAPL']['PG'])

# Question 5
print(calcCorrelations(csvFileName).loc['GOOGL']['PG'])
