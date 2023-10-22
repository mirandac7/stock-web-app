import yfinance as yf
import pandas as pd
import numpy as np
from datetime import date
import dateutil.relativedelta
from bs4 import BeautifulSoup
import requests

def scrape_news(ticker_name):
    link='https://finviz.com/quote.ashx?t=' + ticker_name + '&p=d'
    headers={'User-Agent':'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/107.0.0.0 Safari/537.36'}
    source=requests.get(link,headers=headers).text
    soup=BeautifulSoup(source,"html5lib")
    news_table=soup.find('table',id="news-table")

    parsed_data = []

    for row in news_table.findAll('tr'):
        title = row.text
        date_data = row.td.text.split(' ')
        
        if len(date_data) == 1:
            time = date_data[0]
        else:
            date = date_data[0]
            time = date_data[1]

        parsed_data.append([date, time, title])

    parsed_news = pd.DataFrame(parsed_data, columns=['date', 'time', 'title'])
    #clean up the title 
    parsed_news['Contains Date'] = parsed_news.apply(lambda x: x.date in x.title, axis=1)

    parsed_news['title'] = parsed_news.apply(lambda x: x['title'].replace(x['time'], ''), axis=1)
    parsed_news['title'] = np.where(parsed_news['Contains Date'] == True,
                        parsed_news.apply(lambda x: x['title'].replace(x['date'], ''), axis=1), parsed_news['title'])


    parsed_news = parsed_news.drop('Contains Date', axis=1)
    parsed_news.columns = ['Date', 'Time', 'Title']

    parsed_news['Date'] = pd.to_datetime(parsed_news['Date'])
    parsed_news['Date'] = parsed_news['Date'].dt.strftime('%m/%d/%Y')

    return parsed_news