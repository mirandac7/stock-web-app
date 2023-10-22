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

    # Find all 'td' elements with align='right' (date and time)
    date_time_tags = soup.find_all('td', align='right', width ='130')

    # Find all 'a' elements with class 'tab-link-news' under the 'div' with class 'news-link-container' (news text and URL)
    news_links = soup.select('div.news-link-container a.tab-link-news')

    # Create a list to store the extracted data
    data = []

    # Iterate through the 'td' and 'a' elements to extract the data
    for date_time_tag, news_link in zip(date_time_tags, news_links):
        date_time = date_time_tag.text.strip()    
        news_text = news_link.text
    
        data.append([date_time, news_text])
    parsed_news = pd.DataFrame(data, columns=['Date and Time', 'Title'])
    parsed_news[["Date","Time"]] = parsed_news["Date and Time"].str.split(' ', 1, expand=True)
    parsed_news.drop(columns=["Date and Time"], inplace=True)
    parsed_news['Time'] = np.where(parsed_news['Time'].isnull(),parsed_news['Date'],parsed_news['Time'])
    
    parsed_news['Date'] = np.where((parsed_news['Date'].str.contains('-',na=False))|\
                                    (parsed_news['Date'].str.contains('Today',na=False)),parsed_news['Date'],np.nan)
   
    parsed_news['Date'].fillna(method='ffill', inplace=True)
    today = date.today().strftime('%b-%d-%y')
    parsed_news['Date'] = parsed_news['Date'] .replace('Today', today)
    parsed_news['Date'] = pd.to_datetime(parsed_news['Date'])
    parsed_news['Date'] = parsed_news['Date'].dt.strftime('%m/%d/%Y')
    parsed_news = parsed_news[['Date','Time','Title']]
    return parsed_news