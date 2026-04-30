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
  correlations = correlations[['SPX']]
  correlations.columns = ['Correlation with SPX']
  return correlations

print(calcCorrelations(csvFileName))


# Question 2
print(calcCorrelations(csvFileName).loc['10516']['Correlation with SPX'])
