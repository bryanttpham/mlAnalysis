import csv
import random
from bs4 import BeautifulSoup
import requests
import pandas as pd
import sklearn
from sklearn import linear_model
from sklearn.utils import shuffle
import matplotlib.pyplot as pyplot
import pickle
from matplotlib import style
import numpy as np

data = pd.read_csv("weight-height.csv")

data = data[["Gender","Height","Weight"]]

predict="Weight"

X=np.array(data.drop([predict,"Gender"],1))

y=np.array(data[predict])


x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.1)

best = 0
for _ in range(30):
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.1)
    linear = linear_model.LinearRegression()
    linear.fit(x_train, y_train)
    acc = linear.score(x_test, y_test)
    print("acc: ",acc)

    if acc > best:
        best = acc
        with open("weightHeight.pickle", "wb") as f:
            pickle.dump(linear, f)

pickle_in = open("weightHeight.pickle", "rb")
linear = pickle.load(pickle_in)



print('Co: \n', linear.coef_)
print('Intercept: \n', linear.intercept_)


predictions = linear.predict(x_test)
for x in range(len(predictions)):
    print(predictions[x], x_test[x], y_test[x])


p = 'Height'
style.use("ggplot")
pyplot.scatter(data[p], data["Weight"])
pyplot.scatter(x_test, predictions)
pyplot.xlabel(p)
pyplot.ylabel("Weight")
pyplot.show()


"""
url = "http://socr.ucla.edu/docs/resources/SOCR_Data/SOCR_Data_Dinov_020108_HeightsWeights.html"
data=requests.get(url).text

soup = BeautifulSoup(data, 'html.parser')

table = soup.find('table')

df = pd.DataFrame(columns=['Index','Height','Weight'])

newTable = table.tbody.find_all('tr')[1:]

for row in newTable:
    columns=row.find_all('td')
    if columns != []:
        index = columns[0].text.strip()
        height = columns[1].text.strip()
        weight = columns[2].text.strip()
        df = df.append({'Index': index, 'Height':height, 'Weight':weight},ignore_index=True)
print(df.head())

df.to_csv("baseballwh.csv")
"""
