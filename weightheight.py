import csv
import random
from bs4 import BeautifulSoup
import requests
import pandas as pd

url = "http://socr.ucla.edu/docs/resources/SOCR_Data/SOCR_Data_Dinov_020108_HeightsWeights.html"
data=requests.get(url).text

soup = BeautifulSoup(data, 'html.parser')

table = soup.find('table')

df = pd.DataFrame(columns=['Index','Height','Weight'])


for row in table.tbody.find_all('tr'):
    columns=row.find_all('td')
    if columns != []:
        index = columns[0].text.strip()
        height = columns[1].text.strip()
        weight = columns[2].text.strip()
        df = df.append({'Index': index, 'Height':height, 'Weight':weight},ignore_index=True)
print(df.head())

df.to_csv("C:/Users/bryan/PycharmProjects/tensorEnv/Data Analysis Project")