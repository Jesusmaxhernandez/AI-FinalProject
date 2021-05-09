#!/usr/bin/env python
# coding: utf-8

import quandl
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime

from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression

def predictStockPrice(stockName): 
    #Quandl provides stock data from 1990 to 2018.
    quandl.ApiConfig.api_key = "xNnWAJe4wEoSZT1uJtii"

    dataFrame = quandl.get("WIKI/" + stockName)
    #Looking at adjusted close to account for inflation, stock splits, etc.
    dataFrame = dataFrame[['Adj. Close']]

    #Display data from 2017 - 2018 only.
    dataFrame['Adj. Close'].plot(figsize=(15,6), color ='g')
    plt.legend(loc='upper left')
    plt.xlim(xmin=datetime.date(2017,4,26))
    plt.xlim(xmax=datetime.date(2018,4,26))

    #Predicitng 60 days, first 30 will be compared to the actual price and the
    #other 30 will be predicitng where the price will be going.
    forecast = 60
    dataFrame['Prediction'] = dataFrame[['Adj. Close']].shift(-forecast)

    #Will represent the adjusted close column.
    X = np.array(dataFrame.drop(['Prediction'], 1))
    X = preprocessing.scale(X)
    print(X)

    X_forecast = X[-forecast:]
    X = X[:-forecast]

    y = np.array(dataFrame['Prediction'])
    y = y[:-forecast]


    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    clf = LinearRegression() #Estimator instance
    clf.fit(X_train, y_train)

    confidence = clf.score(X_test, y_test)
    print(confidence) #-1 to 1, closer to 1 the better. (means data is linear)
    forecast_predicted = clf.predict(X_forecast)
    print(forecast_predicted)


    plt.plot(X, y)

    dates = pd.date_range(start="2018-02-26", end="2018-04-26")
    plt.plot(dates, forecast_predicted, color ="y")
    dataFrame['Adj. Close'].plot(figsize=(15,6), color ='g')
    plt.legend(loc='upper left')
    plt.xlim(xmin=datetime.date(2017,4,26))
    plt.xlim(xmax=datetime.date(2018,4,26))
    plt.show()



def main():
    predictStockPrice("AMZN")

main()