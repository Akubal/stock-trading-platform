#!usr/bin/env python3
from flask import Flask, redirect, render_template, request
from flask_restful import Api
from nsetools import Nse
from flask import Flask, render_template, request, url_for, redirect, abort, flash
from flask_sqlalchemy import SQLAlchemy
import sqlite3
import os,math
import random,sys
import smtplib
from datetime import date
from nsepy import get_history
import requests
from plotly import graph_objects as go
from nsetools import Nse
from pandas import DataFrame
import json
import plotly
import ssl
from plotly.subplots import make_subplots
import plotly.io as pio
import pandas as pd
import quandl
import pandas as pd
import numpy as np
import tensorflow as tf
from keras.preprocessing.sequence import TimeseriesGenerator
from nsepy import get_history
from datetime import date
from keras.models import Sequential
from keras.layers import LSTM, Dense
import plotly.graph_objects as go
import model as m
from server import Transactions, UserHoldings, Users, UserTransactions

from numpy.random import seed
seed(1)
tf.random.set_seed(2)


quandl.ApiConfig.api_key = "y7cYXaQgiLtJxY9cWu-J"
pio.templates.default = "plotly_dark"
nse = Nse()
ssl._create_default_https_context = ssl._create_unverified_context
all_stocks = nse.get_stock_codes()

app = Flask(__name__)
username = ""

api = Api(app)
api.add_resource(Users, "/users")
api.add_resource(Transactions, "/transactions")
api.add_resource(UserTransactions, "/transactions/<username>")
api.add_resource(UserHoldings, "/holdings/<username>")


@app.route("/", methods=["GET", "POST"])
@app.route("/login", methods=["GET", "POST"])
def login():
    cannot_login = None
    m.log_out()
    if request.method == "GET":
        return render_template("login.html")
    else:
        submitted_username = request.form["username"]
        submitted_password = request.form["password"]
        result = m.log_in(submitted_username, submitted_password)
        if result:
            return redirect("/home")
        else:
            cannot_login = True
            return render_template("login.html", cannot_login=cannot_login)


@app.route("/create", methods=["GET", "POST"])
def create():
    cannot_create = None
    if request.method == "GET":
        return render_template("create.html")
    else:
        submitted_username = request.form["username"]
        submitted_password = request.form["password"]
        submitted_funds = request.form["funds"]
        result = m.create(submitted_username, submitted_password, submitted_funds)
        if result:
            return redirect("/")
        else:
            cannot_create = True
            return render_template("create.html", cannot_create=cannot_create)


@app.route("/dashboard", methods=["GET", "POST"])
def dashboard():
    current_user = m.current_user()
    if request.method == "GET":
        if current_user == "randomuser":
            return redirect("/")
        else:
            m.update_holdings()
            cu = current_user
            # pnl = m.calculate_p_and_l(username)
            user_holdings = m.display_user_holdings()
            # holdings = pd.DataFrame(user_holdings)
            user_transactions = m.display_user_transactions()
            current_bal = m.cur_bal()
            current_bal = str(current_bal)
            punc = '''!()-[]{};:'"\, <>/?@#$%^&*_~'''
            for ele in current_bal:
                if ele in punc:
                    current_bal = current_bal.replace(ele, "")
            return render_template(
                "dashboard.html", position_list=user_holdings, result=user_transactions, user=cu, bal=current_bal
            )
    else:
        return render_template("dashboard.html", result=None)


@app.route("/trade", methods=["GET", "POST"])
def trade():
    current_user = m.current_user()
    if request.method == "GET":
        if current_user == "randomuser":
            return redirect("/")
        else:
            return render_template("trade.html", user=current_user)
    elif request.method == "POST":
        try:
            submitted_symbol = request.form["ticker_symbol"].upper()
            submitted_volume = request.form["number_of_shares"]
            submitted_volume = int(submitted_volume)
            confirmation_message, transaction = m.buy(
                username, submitted_symbol, submitted_volume
            )
            if submitted_volume == 1:
                result = "You bought {} share of {}.".format(
                    submitted_volume, submitted_symbol
                )
            else:
                result = "You bought {} shares of {}.".format(
                    submitted_volume, submitted_symbol
                )
            m.update_holdings()
            if confirmation_message:
                m.buy_db(transaction)
                return render_template("trade.html", result=result, user=current_user)
            else:
                return render_template("trade.html", user=current_user)
        except Exception:
            submitted_symbols = request.form["ticker_symb"].upper()
            submitted_volumes = request.form["number_shares"]
            submitted_volumes = int(submitted_volumes)
            confirmation_message, transaction = m.sell(
                username, submitted_symbols, submitted_volumes
            )
            if submitted_volumes == 1:
                results = "You sold {} share of {}.".format(
                    submitted_volumes, submitted_symbols
                )
            else:
                results = "You sold {} shares of {}.".format(
                    submitted_volumes, submitted_symbols
                )
            m.update_holdings()
            if confirmation_message:
                m.sell_db(transaction)
                return render_template("trade.html", results=results, user=current_user)
            else:
                return render_template("trade.html", cannot_sell=True, user=current_user)

@app.route("/home", methods=["GET", "POST"])
def home():
    current_user = m.current_user()
    if request.method == "GET":
        if current_user == "randomuser":
            return redirect("/")
        else:
            quotes = []
            stocknames = ['TATAMOTORS', 'RELIANCE', 'SBIN', 'ASIANPAINT', 'HDFCBANK', 'ICICIBANK', 'INFY', 'TCS', 'JSWSTEEL',
                          'LT', 'LTI', 'NDTV', 'PFIZER', 'WHIRLPOOL', 'AMBUJACEM', 'APOLLO', 'ASTRAZEN',
                          'BANKBARODA', 'BHARTIARTL', 'BRITANNIA', 'CENTRALBK', 'CANBK']
            for i in stocknames:
                q = nse.get_quote(i)
                quotes.append(i + ": ₹" + str(q['lastPrice']))
            bar = dailygainersplot()
            data = ('  '.join(quotes))
            return render_template("home.html", plot=bar, data=data)



@app.route("/search", methods=["GET", "POST"])
def search():
    current_user = m.current_user()
    if request.method == "GET":
        if current_user == "randomuser":
            return redirect("/")
        else:
            return render_template("search.html", user=current_user)
    elif request.method == "POST":
        try:
            submitted_company_name = request.form["company_name"]
            submitted_company_name = submitted_company_name.capitalize()
            ticker_symboll = m.lookup_ticker_symbol(submitted_company_name)
            if ticker_symboll:
                result = "The ticker symbol for {} is {}.".format(
                    submitted_company_name, ticker_symboll
                )
                return render_template("search.html", resultthree=result, user=current_user)
            else:
                return render_template("search.html", company_dne=True, user=current_user)

        except Exception:
            submitted_symbol = request.form["ticker_symbol"]
            submitted_symbol = submitted_symbol.upper()
            price = m.quote_last_price(submitted_symbol)
            results = "The last price of {} is ₹{}.".format(submitted_symbol, price)
            return render_template("search.html", resultfour=results, user=current_user)


def dailygainersplot():
    losecolors = ['Brown','Crimson','DarkRed','FireBrick','IndianRed','LightCoral','Maroon','Red','Sienna', 'Tomato']
    gaincolors = ['DarkGreen','DarkOliveGreen','ForestGreen','Green','LightGreen','LawnGreen','LimeGreen','MediumSeaGreen','OliveDrab','SeaGreen']
    gain = nse.get_top_gainers()
    df = DataFrame(gain, columns=['symbol', 'netPrice'])
    tradgain = DataFrame(gain, columns =['symbol', 'tradedQuantity'])
    losers = nse.get_top_losers()
    df2 = DataFrame(losers, columns=['symbol', 'netPrice'])
    tradlose = DataFrame(losers, columns =['symbol', 'tradedQuantity'])
    df3 = quandl.get("BSE/SENSEX", start_date="2020-06-01", end_date="2021-03-19")
    df4 = get_history(symbol="NIFTY50", start=date(2020, 6, 1), end=date.today())
    df3['Date'] = df3.index
    df4['Date'] = df4.index
    df3['Date'] = pd.to_datetime(df3['Date'])
    df3.set_axis(df3['Date'], inplace=True)
    df4['Date'] = pd.to_datetime(df4['Date'])
    df4.set_axis(df4['Date'], inplace=True)
    trace = make_subplots(specs=[[{"type": "xy"},{"type": "xy"}],[{"type": "xy"},{"type": "xy"}],[{"type": "domain"},{"type": "domain"}]],rows=3, cols=2, horizontal_spacing=0.2, subplot_titles=("NIFTY 50","BSE SENSEX","Daily Gainers", "Daily Losers","Gainers Traded Quantity", "Losers Traded Quantity"))
    trace.add_trace(
        go.Scatter(
            name='Nifty-50',
            x=df4.Date,
            y=df4.Close,
            marker_color='magenta'),

        row=1,
        col=1,
    )
    trace.add_trace(
        go.Scatter(
            name='Sensex',
            x=df3.Date,
            y=df3.Close,
            marker_color='blue'),

        row=1,
        col=2,
    )

    trace.add_trace(
        go.Bar(
            name = 'Gainers',
            x=df.symbol,
            y=df.netPrice,
            marker_color='LimeGreen'),
        row = 2,
        col =1,
    )
    trace.add_trace(
        go.Bar(
            name = 'Losers',
            x=df2.symbol,
            y=df2.netPrice,
            marker_color='red'),
        row=2,
        col=2,
    )
    trace.add_trace(
        go.Pie(
            labels=tradgain.symbol,
            values=tradgain.tradedQuantity,
            hole=0.25,
            name='Daily Gainers Trading Volume',
            marker_colors= gaincolors),
        row=3,
        col=1,
    )
    trace.add_trace(
        go.Pie(
            labels=tradlose.symbol,
            values=tradlose.tradedQuantity,
            hole=0.25,
            name='Daily Losers Trading Volume',
            marker_colors=losecolors),
        row=3,
        col=2
    )

    trace.update_layout(height=1350, width=1200, title_text="NSE and BSE Overview", title_x=0.5, showlegend=False,)
    trace.update_xaxes(title_text="Companies", row=2, col=1)
    trace.update_xaxes(title_text="Companies", row=2, col=2)
    trace.update_xaxes(title_text="Date", row=1, col=1)
    trace.update_xaxes(title_text="Date", row=1, col=2)
    trace.update_yaxes(title_text="Net Price Gain", row=2, col=1)
    trace.update_yaxes(title_text="Net Price Loss", row=2, col=2)
    trace.update_yaxes(title_text="Price", row=1, col=1)
    trace.update_yaxes(title_text="Price", row=1, col=2)


    graphJSON = json.dumps(trace, cls=plotly.utils.PlotlyJSONEncoder)

    return graphJSON

@app.route('/tata', methods = ['POST', 'GET'])
def tata():
    namestock = 'TATAMOTORS'
    name = 'Tata Motors'
    bar = lstm(namestock=namestock, name = name)
    tatasent = pd.read_excel(('./tata.xlsx'), engine='openpyxl')
    pol = tatasent['probability'].tolist()
    avgpol = sum(pol) / len(pol)
    # print(pol)
    print(avgpol)
    if 0 < avgpol < 0.5:
        senti = "Tata Motor's market sentiment is Moderately Positive, which indicates a small rise in it's stock price."
    elif 0.5 < avgpol < 1:
        senti = "Tata Motor's market sentiment is Positive, which indicates a rise in it's stock price."
    elif -0.5 < avgpol < 0:
        senti = "Tata Motor's market sentiment is Moderately Negative, which indicates a small drop in it's stock price."
    else:
        senti = "Tata Motor's market sentiment is Negative, which indicates a fall in it's stock price."

    return render_template('tata.html', plot=bar, data=senti)

@app.route('/hdfc', methods = ['POST', 'GET'])
def hdfc():
    namestock = 'HDFCBANK'
    name = 'HDFC Bank'
    bar = lstm(namestock=namestock, name = name)
    hdfcsent = pd.read_excel(('./hdfc.xlsx'), engine='openpyxl')
    pol = hdfcsent['probability'].tolist()
    avgpol = sum(pol) / len(pol)
    # print(pol)
    print(avgpol)
    if 0 < avgpol < 0.5:
        senti = "HDFC's market sentiment is Moderately Positive, which indicates a small rise in it's stock price."
    elif 0.5 < avgpol < 1:
        senti = "HDFC's market sentiment is Positive, which indicates a rise in it's stock price."
    elif -0.5 < avgpol < 0:
        senti = "HDFC's market sentiment is Moderately Negative, which indicates a small drop in it's stock price."
    else:
        senti = "HDFC's market sentiment is Negative, which indicates a fall in it's stock price."

    return render_template('hdfc.html', plot=bar, data=senti)

@app.route('/reliance', methods = ['POST', 'GET'])
def reliance():
    namestock = 'RELIANCE'
    name = 'Reliance'
    bar = lstm(namestock=namestock, name = name)
    relsent = pd.read_excel(('./reliance.xlsx'), engine='openpyxl')
    pol = relsent['probability'].tolist()
    avgpol = sum(pol) / len(pol)
    # print(pol)
    print(avgpol)
    if 0 < avgpol < 0.5:
        senti = "Reliance's market sentiment is Moderately Positive, which indicates a small rise in it's stock price."
    elif 0.5 < avgpol < 1:
        senti = "Reliance's market sentiment is Positive, which indicates a rise in it's stock price."
    elif -0.5 < avgpol < 0:
        senti = "Reliance's market sentiment is Moderately Negative, which indicates a small drop in it's stock price."
    else:
        senti = "Reliance's market sentiment is Negative, which indicates a fall in it's stock price."

    return render_template('reliance.html', plot=bar, data=senti)

@app.route('/sbi', methods = ['POST', 'GET'])
def sbi():
    namestock = 'SBIN'
    name = 'SBI Bank'
    bar = lstm(namestock=namestock, name = name)
    sbisent = pd.read_excel(('./sbi.xlsx'), engine='openpyxl')
    pol = sbisent['probability'].tolist()
    avgpol = sum(pol) / len(pol)
    # print(pol)
    print(avgpol)
    if 0 < avgpol < 0.5:
        senti = "SBI's market sentiment is Moderately Positive, which indicates a small rise in it's stock price."
    elif 0.5 < avgpol < 1:
        senti = "SBI's market sentiment is Positive, which indicates a rise in it's stock price."
    elif -0.5 < avgpol < 0:
        senti = "SBI's market sentiment is Moderately Negative, which indicates a small drop in it's stock price."
    else:
        senti = "SBI's market sentiment is Negative, which indicates a fall in it's stock price."

    return render_template('sbi.html', plot=bar, data=senti)

@app.route('/asianp', methods = ['POST', 'GET'])
def asianp():
    namestock = 'ASIANPAINT'
    name = 'Asian Paint'
    bar = lstm(namestock=namestock, name = name)
    assent = pd.read_excel(('./asianp.xlsx'), engine='openpyxl')
    pol = assent['probability'].tolist()
    avgpol = sum(pol) / len(pol)
    # print(pol)
    print(avgpol)
    if 0 < avgpol < 0.5:
        senti = "Asian Paints's market sentiment is Moderately Positive, which indicates a small rise in it's stock price."
    elif 0.5 < avgpol < 1:
        senti = "Asian Paint's market sentiment is Positive, which indicates a rise in it's stock price."
    elif -0.5 < avgpol < 0:
        senti = "Asian Paint's market sentiment is Moderately Negative, which indicates a small drop in it's stock price."
    else:
        senti = "Asian Paint's market sentiment is Negative, which indicates a fall in it's stock price."
    return render_template('asianp.html', plot=bar, data=senti)

@app.route('/icici', methods = ['POST', 'GET'])
def icici():
    namestock = 'ICICIBANK'
    name = 'ICICI Bank'
    bar = lstm(namestock=namestock, name = name)
    isent = pd.read_excel(('./icici.xlsx'), engine='openpyxl')
    pol = isent['probability'].tolist()
    avgpol = sum(pol) / len(pol)
    # print(pol)
    print(avgpol)
    if 0 < avgpol < 0.5:
        senti = "ICICI's market sentiment is Moderately Positive, which indicates a small rise in it's stock price."
    elif 0.5 < avgpol < 1:
        senti = "ICICI's market sentiment is Positive, which indicates a rise in it's stock price."
    elif -0.5 < avgpol < 0:
        senti = "ICICI's market sentiment is Moderately Negative, which indicates a small drop in it's stock price."
    else:
        senti = "ICICI's market sentiment is Negative, which indicates a fall in it's stock price."
    return render_template('icici.html', plot=bar, data=senti)

@app.route('/infy', methods = ['POST', 'GET'])
def infy():
    namestock = 'INFY'
    name = 'Infosys Limited'
    bar = lstm(namestock=namestock, name = name)
    infysent = pd.read_excel(('./infy.xlsx'), engine='openpyxl')
    pol = infysent['probability'].tolist()
    avgpol = sum(pol) / len(pol)
    # print(pol)
    print(avgpol)
    if 0 < avgpol < 0.5:
        senti = "Infosys' market sentiment is Moderately Positive, which indicates a small rise in it's stock price."
    elif 0.5 < avgpol < 1:
        senti = "Infosys' market sentiment is Positive, which indicates a rise in it's stock price."
    elif -0.5 < avgpol < 0:
        senti = "Infosys' market sentiment is Moderately Negative, which indicates a small drop in it's stock price."
    else:
        senti = "Infosys' market sentiment is Negative, which indicates a fall in it's stock price."
    return render_template('infy.html', plot=bar, data=senti)

@app.route('/tcs', methods = ['POST', 'GET'])
def tcs():
    namestock = 'TCS'
    name = 'Tata Consultancy Services Limited'
    bar = lstm(namestock=namestock, name = name)
    tcssent = pd.read_excel(('./tcs.xlsx'), engine='openpyxl')
    pol = tcssent['probability'].tolist()
    avgpol = sum(pol) / len(pol)
    # print(pol)
    print(avgpol)
    if 0 < avgpol < 0.5:
        senti = "TCS' market sentiment is Moderately Positive, which indicates a small rise in it's stock price."
    elif 0.5 < avgpol < 1:
        senti = "TCS' market sentiment is Positive, which indicates a rise in it's stock price."
    elif -0.5 < avgpol < 0:
        senti = "TCS' market sentiment is Moderately Negative, which indicates a small drop in it's stock price."
    else:
        senti = "TCS' market sentiment is Negative, which indicates a fall in it's stock price."
    return render_template('tcs.html', plot=bar, data=senti)

def lstm(namestock, name):
    df = get_history(symbol=namestock, start=date(2012, 1, 1), end=date.today())
    df['Date'] = df.index
    print(df.info())
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_axis(df['Date'], inplace=True)
    df.drop(columns=['Open', 'High', 'Low', 'Volume'], inplace=True)
    close_data = df['Close'].values
    last_element = close_data[-1]
    print('The last was: ' + str(last_element))
    close_data = close_data.reshape((-1, 1))

    split_percent = 0.80
    split = int(split_percent * len(close_data))

    close_train = close_data[:split]
    close_test = close_data[split:]

    date_train = df['Date'][:split]
    date_test = df['Date'][split:]

    print(len(close_train))
    print(len(close_test))

    look_back = 10

    train_generator = TimeseriesGenerator(close_train, close_train, length=look_back, batch_size=20)
    test_generator = TimeseriesGenerator(close_test, close_test, length=look_back, batch_size=1)

    model = Sequential()
    model.add(
        LSTM(10,
             activation='relu',
             input_shape=(look_back, 1))
    )
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')

    num_epochs = 25
    model.fit(train_generator, epochs=num_epochs, verbose=1)


    prediction = model.predict(test_generator)
    prepredict = model.predict(train_generator)

    close_data = close_data.reshape((-1))

    def predict(num_prediction, model):
        prediction_list = close_data[-look_back:]

        for _ in range(num_prediction):
            x = prediction_list[-look_back:]
            x = x.reshape((1, look_back, 1))
            out = model.predict(x)[0][0]
            prediction_list = np.append(prediction_list, out)
        prediction_list = prediction_list[look_back - 1:]
        print('The forecasted price of ' + namestock + ' is: ' + str(prediction_list[-1]))
        global forecastfinal
        forecastfinal = prediction_list[-1]

        return prediction_list

    def predict_dates(num_prediction):
        last_date = df['Date'].values[-1]
        prediction_dates = pd.date_range(last_date, periods=num_prediction + 1).tolist()
        return prediction_dates

    num_prediction = 30
    forecast = predict(num_prediction, model)
    forecast_dates = predict_dates(num_prediction)

    close_train = close_train.reshape((-1))
    close_test = close_test.reshape((-1))
    prediction = prediction.reshape((-1))
    forecast = forecast.reshape((-1))
    prepredict = prepredict.reshape((-1))

    trace1 = go.Scatter(
        x=date_train,
        y=close_train,
        mode='lines',
        name='Data'
    )
    trace2 = go.Scatter(
        x=date_test,
        y=prediction,
        mode='lines',
        name='Prediction'
    )
    trace3 = go.Scatter(
        x=date_test,
        y=close_test,
        mode='lines',
        name='Ground Truth'
    )
    trace4 = go.Scatter(
        x=date_train,
        y=prepredict,
        mode='lines',
        name='Training prediction'
    )
    trace5 = go.Scatter(
        x=forecast_dates,
        y=forecast,
        mode='lines',
        name='Forecast'
    )
    layout = go.Layout(
        title="Stock Price",
        xaxis={'title': "Date"},
        yaxis={'title': "Close"}
    )
    fig = go.Figure(data=[trace1, trace2, trace3, trace4, trace5], layout=layout)
    fig.update_layout(height=800, width=1200, title_text=name, title_x= 0.5)
    fig.update_layout(
        xaxis=dict(
            rangeslider=dict(
                visible=True
            ),
            type="date"
        )

    )

    graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

    return graphJSON



if __name__ == "__main__":
    app.run(debug=True)
