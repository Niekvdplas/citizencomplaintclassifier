from sklearn.model_selection import train_test_split
import pandas as pd
import random


fields = ['category', 'description']
data = pd.read_csv('SAMPLE_DATA', usecols=fields) #Input your data file here.
data = data.sample(n=100)

data.to_csv('humansample.csv')
