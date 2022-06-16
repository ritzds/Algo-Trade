"""AlgoTradeENV is a virtual env I was using as future updates should not affect our written code.
    It is recommended to use it. But as of now code will just fine without it also.

    dbcontext is the database file. which we have to run to create a database in a new machine.
    first uncomment(ctrl + K,ctrl+L) all the lines which are together(without any spaces between them) then run the file,
    then comment those uncommented lines again. and repeat the cycle.
    more details on dbcontext.

    To install all the required packages you can utilize the requirements.txt file create.

    Commandline process the create a virtualenv and install all the dependencies
    virtualenv <environment_name>                                   # Create a virtual environment
    cd <directory>/<environment_name>\Scripts\activate.bat          # Activate the virtual environment
    pip install -r requirements.txt # Install the dependencies      # to installall the packages from requirements.txt
"""


from flask import Flask,render_template,url_for,redirect,request,flash  # just install flask as of now
import os
import sqlite3
import pickle as pkl
import yfinance as yf
import pandas as pd
import numpy as np
import nltk
import itertools
from bs4 import BeautifulSoup
from urllib.request import urlopen, Request
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from datetime import datetime,timedelta
import datetime as dt
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torchvision
import torchaudio
import re
import keras
import tensorflow as tf
import pandas_datareader as pdr
from tensorflow.keras.models import load_model
from nltk.stem.porter import PorterStemmer
from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.layers import Embedding
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense
from nltk.corpus import stopwords

nltk.download('vader_lexicon')


loc = os.path.dirname(os.path.abspath(__file__))

def price_fetch():
    """
    It takes the stock price data from yahoo finance and converts it into a list of dates and a list of
    prices
    :return: A tuple of two lists.
    """
    df = yf.download(tickers="GOOGL",period='5y',interval='1mo').reset_index()
    df['Date']=df['Date'].map(str)
    df['Open']=df['Open'].astype(int)
    arr=[]
    for i in df["Date"]:
        s=""
        for j in i:
            s1 = i[2:4]
            s2 = i[5:7]
            s=s2+"-"+s1
        arr.append(s)
    df["monthyear"] = arr
    date= df["monthyear"].tolist()
    price=df["Close"].tolist()

    return date,price


def name_fetch(EID):
    """
    It takes an email ID as input and returns the corresponding username

    :param EID: Email ID of the user
    :return: The name of the user.
    """
    sqlconnection = sqlite3.Connection("AlgoTrade.db")
    cursor = sqlconnection.cursor()
    query1 = "SELECT * FROM User WHERE email =\'" + EID+"\'";
    name = cursor.execute(query1)
    username = name.fetchone()[1]
    return str(username)


def getSIA(text):
    """
    It takes a string of text as input, and returns a dictionary of scores for each of the four
    sentiment categories
   
    :param text: The text you want to analyze
    :return: A dictionary with the following keys:
    """
    sia = SentimentIntensityAnalyzer()
    sentiment = sia.polarity_scores(text)
    return sentiment


def news_fetch():
    """
    It takes the news headlines from the last two days, runs them through a sentiment analysis model,
    and returns the headlines, the sentiment scores, and the sentiment labels
    :return: title,compound,sentiment,sample_df
    """
    news_tables={}
    finviz_url = 'https://finviz.com/quote.ashx?t='
    ticker = 'GOOGL'
    url = finviz_url + ticker
    req = Request(url=url, headers={'user-agent': 'my-app'})
    response = urlopen(req)
    html = BeautifulSoup(response, features='html.parser')

    news_table = html.find(id='news-table')
    news_tables[ticker] = news_table

    parsed_data = []
    for ticker, news_table in news_tables.items():
        for row in news_table.findAll('tr'):
            titles = row.a.text
            date_data = row.td.text.split(' ')
            if len(date_data) == 1:
                time = date_data[0]
            else:
                date = date_data[0]
                time = date_data[1]
            parsed_data.append([ticker, date, time, titles])

    news = pd.DataFrame(parsed_data, columns=['Ticker', 'Date', 'Time', 'Titles'])
    sample_df = news
    news['Date'] = pd.to_datetime(news['Date'])
    compound = []
    for i in range(0,len(news['Titles'])):
        SIA = getSIA(news['Titles'][i])
        compound.append(SIA['compound'])
    news["compound"] = compound

    news=bert_score(news)


    end_date = datetime.today()
    start_date = end_date - timedelta(days=2)
    after_start_date = news['Date'] >= start_date
    before_end_date = news['Date'] <= end_date
    between_two_dates = after_start_date & before_end_date
    news = news.loc[between_two_dates]


    title = news["Titles"].tolist()
    compound = news["compound"].tolist()
    sentiment = news["sentiment"].tolist()

    return title,compound,sentiment,sample_df


def email_check(EID):
    """
    It checks if the email ID is already registered in the database.

    :param EID: Email ID
    :return: a boolean value.
    """
    sqlconnection = sqlite3.Connection("AlgoTrade.db")
    cursor = sqlconnection.cursor()
    query1 = "SELECT * FROM User WHERE email =\'" + EID+"\'";
    rows = cursor.execute(query1)
    rows = rows.fetchall()

    if len(rows)>0:
        return False
    return True


def sentiment_score(headlines,tokenizer,model):
    """
    1. Tokenize the headline
    2. Pass the tokens to the model
    3. Return the sentiment score

    Let's test it out on a few headlines

    :param headlines: The text you want to analyze
    :param tokenizer: The tokenizer we created earlier
    :param model: The model we're using to predict sentiment
    :return: The sentiment score of the headline.
    """
    tokens = tokenizer.encode(headlines, return_tensors='pt')
    result = model(tokens)
    return int(torch.argmax(result.logits))+1


def bert_score(news):
    """
    It takes a dataframe of news articles as input, and returns a dataframe of news articles with a
    sentiment score for each article

    :param news: the dataframe containing the news titles
    :return: The sentiment score of the news titles.
    """
    tokenizer = AutoTokenizer.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')
    model = AutoModelForSequenceClassification.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')
    news['sentiment'] = news['Titles'].apply(lambda x: sentiment_score(x,tokenizer,model))
    return news


def sentiment_analysis(final):
    """
    It takes the dataframe as input, preprocesses it, and then uses the model to predict the sentiment
    of the titles

    :param final: The dataframe that contains the titles of the articles
    :return: The sentiment of the title.
    """

    from tensorflow.keras.models import load_model
    new_model = load_model('SentimentAnalysisModel.h5')
    voc_size=5000
    ps = PorterStemmer()
    corpus = []
    for i in range(0, len(final)):
        review = re.sub('[^a-zA-Z]', ' ', final['Titles'][i])
        review = review.lower()
        review = review.split()

        review = [ps.stem(word) for word in review if not word in stopwords.words('english')]
        review = ' '.join(review)
        corpus.append(review)

    onehot_repr=[one_hot(words,voc_size)for words in corpus]
    onehot_repr

    sent_length=20
    embedded_docs=pad_sequences(onehot_repr,padding='pre',maxlen=sent_length)
    #print(embedded_docs)

    X_test=np.array(embedded_docs)

    bfr_predict=new_model.predict(X_test)
    y_pred = np.argmax(bfr_predict,axis=1)

    return y_pred

def lstm_predict():
    """
    It takes the last 100 days of Google stock data, transforms it, and then uses the LSTM model to
    predict the next 30 days of stock prices
    :return: The last value of the list lst.
    """


    scaler = pkl.load(open('scaler.pkl','rb'))
    model = load_model('final_model.h5')
    end = dt.datetime.now()
    start = end - dt.timedelta(days = 200)
    data_google = pdr.get_data_yahoo('GOOGL',start,end)
    df = data_google.tail(100)
    lstm_df = np.array(df["Adj Close"])
    lstm_df = lstm_df.reshape(100,1)
    lstm_df = scaler.fit_transform(lstm_df.reshape(-1,1))
    x_input=lstm_df.reshape(1,-1)

    temp_date = [vals for vals in range(1,31)]  #For getting a hard coded list of dates for the next month predicted data


    temp_input=list(x_input)
    temp_input=temp_input[0].tolist()

    lst_output=[]
    n_steps=100
    i=0
    while(i<30):
        if(len(temp_input)>100):
            x_input=np.array(temp_input[1:])
            x_input=x_input.reshape(1,-1)
            x_input = x_input.reshape((1, n_steps, 1))

            yhat = model.predict(x_input, verbose=0)

            temp_input.extend(yhat[0].tolist())
            temp_input=temp_input[1:]

            lst_output.extend(yhat.tolist())
            i=i+1
        else:
            x_input = x_input.reshape((1, n_steps,1))
            yhat = model.predict(x_input, verbose=0)

            temp_input.extend(yhat[0].tolist())

            lst_output.extend(yhat.tolist())
            i=i+1
    lst = scaler.inverse_transform(lst_output)

    return lst,temp_date



def gross_profit():
    """
    It takes the financials of the company Google, transposes the dataframe, creates a new column called
    Gross Profit, divides the values in the Gross Profit column by 1,000,000,000, and then returns a
    list of the values in the Gross Profit column
    :return: A list of gross profit values for Google.
    """
    googl = yf.Ticker("GOOGL")
    df = googl.financials
    googlT = df.transpose()
    googlT["Gross Profit"]= googlT["Gross Profit"]/1000000000
    gp= googlT["Gross Profit"].tolist()

    return gp

def net_income():
    """
    It takes the net income of Google from the Yahoo Finance API and returns a list of the net income
    for the past 5 years
    :return: A list of net income values for Google.
    """
    googl = yf.Ticker("GOOGL")
    df = googl.financials
    googlT = df.transpose()
    googlT["Net Income"]= googlT["Net Income"]/1000000000
    ni = googlT["Net Income"].tolist()

    return ni

def total_revenue():
    """
    It takes the total revenue of Google from the year 2015 to 2019 and returns it as a list.
    :return: A list of total revenue for each year.
    """
    googl = yf.Ticker("GOOGL")
    df = googl.financials
    googlT = df.transpose()
    googlT["Total Revenue"]= googlT["Total Revenue"]/1000000000
    tr= googlT["Total Revenue"].tolist()

    return tr

def research_development():
    """
    The function takes the financials of the company Google and returns a list of the research and
    development expenses for the past 5 years
    :return: A list of the Research and Development expenses for Google from the last 10 years.
    """
    googl = yf.Ticker("GOOGL")
    df = googl.financials
    googlT = df.transpose()
    googlT["Research Development"]= googlT["Research Development"]/1000000000
    rd= googlT["Research Development"].tolist()

    return rd




app=Flask(__name__)
app.secret_key = "super secret key"


""" only get method will be redirected to the method """
@app.route('/')
def homepage():
    return render_template('Index.html')


"""
    It takes the email and password from the user and checks if the email and password are present in
    the database. If they are, it fetches the name of the user, the stock price, the next day's stock
    price, the news headlines, the sentiment analysis of the news headlines and the gross profit, total
    revenue, net income and research and development of the company.
    :return: the home page of the website.
"""
@app.route('/signin',methods = ["GET","POST"])
def signin():
    if request.method == "POST":
        EID = request.form['email']
        PW = request.form['password']

        sqlconnection = sqlite3.Connection("AlgoTrade.db")
        cursor = sqlconnection.cursor()
        query = "SELECT email,password From User WHERE email = \'" + EID+"\'" +" AND password=\'"+PW +"\'"
        rows = cursor.execute(query)
        rows = rows.fetchall()

        if len(rows) == 1:
            username = name_fetch(EID)

            date, price = price_fetch()

            #next_day_price = lstm_predict()

            
            temp_price,temp_date = lstm_predict()
            temp_price = [temp_price.tolist()[i][0] for i in range(len(temp_price))]

            last_day_pred_price = temp_price[-1]

            title,compound,sentiment,sample_df = news_fetch()

            pos_neg = sentiment_analysis(sample_df)
            if pos_neg[0] == 1:
                bull_bear="Bullish"
            if pos_neg[0] == 0:
                bull_bear="Bearish"

            graph_index = ['2021','2020','2019','2018']
            gp = gross_profit()
            tr = total_revenue()
            ni = net_income()
            rd = research_development()

            return render_template('Home.html',
            labels=date,
            values=price,
            user = str(username),headline_compound_sentiment = zip(title,compound,sentiment),
            bull_bear=bull_bear,
            last_day_pred_price = last_day_pred_price,
            temp_date = temp_date,
            ritz = temp_price,
            graph_index = graph_index,
            gp = gp , 
            tr = tr,
            ni = ni,
            rd = rd)
        else:
            flash("Email Id or Password Incorrect.")
            return redirect("/")
    else:
        return render_template("signin.html")



"""
    If the request method is POST, then get the name, email and password from the form, check if the
    email is already registered, if not, then insert the name, email and password into the database and
    redirect to the home page
    :return: the rendered template of signup.html
"""
@app.route("/signup",methods = ["GET","POST"])
def signup():
    if request.method == "POST":
        nName = request.form.get("nname")
        nEmail = request.form.get("nemail")
        nPassword = request.form.get("npassword")

        if email_check(nEmail):
            conn = sqlite3.Connection("AlgoTrade.db")
            cur = conn.cursor()
            query = "INSERT INTO User (name,email,password) VALUES ( '{n}','{e}','{p}')".format(n = nName,e = nEmail,p = nPassword)
            cur.execute(query)
            conn.commit()
            return redirect("/")
        else:
            flash("This email id is already registered. Please Log In or signup with different email id")
            return redirect("/")

    return render_template("signup.html")

"""
    It takes the data from the two functions and renders it to the html page.
    :return: the rendered template of the Home.html file.
"""
@app.route('/home')
def home():
    date, price = price_fetch()
    labels = [rows for rows in date]
    values = [rows for rows in price]

    title,compound = news_fetch()

    return render_template('Home.html', labels = date, values = price,user = str(username),headline_compound = zip(title,compound))


"""
    The function about() is a route that renders the About.html template
    :return: the template 'About.html'
"""
@app.route('/about')
def about():
    return render_template('About.html')

"""
    It renders the Contact.html page.
    :return: the render_template function.
"""
@app.route('/contact')
def contact():
    return render_template('Contact.html')


if __name__ == '__main__':
    app.run(host = "0.0.0.0",port = 5000)
