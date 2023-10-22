import yfinance as yf
import streamlit as st
import pandas as pd
import numpy as np
from datetime import date,datetime
from pandas.tseries.offsets import BDay
import dateutil.relativedelta
import plotly.express as px
from news_scrapper import *
from sentiment_analysis import *
from plotly import graph_objs as go
from PIL import Image


from urllib.request import urlopen, Request
from bs4 import BeautifulSoup
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt
from plotly.subplots import make_subplots

#import s&p500 stocks 
sp_500 = pd.read_csv(r'/Users/mirandacheng7/Desktop/Study/Harrisburg University/Research Methodology & Writing/Applied Project/data/S&P500.csv')


ticker_name = st.session_state["my_input"]
company_name = sp_500[sp_500['Symbol'] == ticker_name].iloc[0]['Name']
sector_name = sp_500[sp_500['Symbol'] == ticker_name]['Sector']



st.write(f"""
# Stock Price Analysis 📈
##### Company: {ticker_name} - {company_name}
This page allows you to view stock price, news and sentiment analysis for an individual stock.
""")

image = Image.open('images/imageforapp2.jpg')
st.sidebar.image(image, caption='',
                 width=400, use_column_width='auto')

st.sidebar.header('Stock Price Analysis')

start = st.date_input('Start',value=pd.to_datetime('2022-01-01'))
end = st.date_input('End', value = pd.to_datetime('today'))


tickerData = yf.Ticker(ticker_name)
#get the historical prices for this ticker
tickerDf = tickerData.history(period='1d', start=start, end=end)
tickerDf = tickerDf.reset_index()

@st.cache
def load_data(ticker):
    data = yf.download(ticker, start=start, end =end)
    data.reset_index(inplace=True)
    return data

st.subheader('1.Data Loading 🏋️')
	
data_load_state = st.text('Loading data...')
data = load_data(ticker_name)
data_load_state.text('Loading data... done!')

st.subheader('Raw data')
st.markdown('The last 5 days of stock closing price ')
st.write(data.tail())

st.subheader('2.Summary')

def create_summary():
    stock_table = {}
    # stock_table['52 Week High'] = tickerData.info['fiftyTwoWeekHigh']
    stock_table['Last Date']  = [end]
    stock_table['Last Price'] = ['{:.2f}'.format(tickerData.fast_info['lastPrice'])]
    stock_table['52 Week High']  = [tickerData.info['fiftyTwoWeekHigh']]
    stock_table['P/E Ratio'] = [tickerData.info['forwardPE']]
    stock_table['Market Cap'] = ['{:20,.0f}'.format(tickerData.info['marketCap'])]
    stock_table['Shares'] = ['{:20,.0f}'.format(tickerData.fast_info['shares'])]
    stock_table['Last Volume'] = ['{:20,.0f}'.format(tickerData.fast_info['lastVolume'])]
    # stock_table['Three-month Average Volume'] = ['{:20,.0f}'.format(tickerData.fast_info['threeMonthAverageVolume'])]
    summary = pd.DataFrame(stock_table)
    return summary

# CSS to inject contained in a string
hide_table_row_index = """
            <style>
            thead tr th:first-child {display:none}
            tbody th {display:none}
            </style>
            """
# Inject CSS with Markdown
st.markdown(hide_table_row_index, unsafe_allow_html=True)

df_stock = create_summary()
st.table(df_stock)

st.subheader('3.Trend Analysis')

st.write(f"""
#### Closing Price by Date from {start} to {end}
""")

fig1 = px.line(tickerDf, x='Date', y='Close')
fig1.update_layout(
    width=900,
    height=500,)

st.plotly_chart(fig1)

# st.line_chart(tickerDf.Close)

st.write("""
#### Closing Price and Moving Averages
""")
ma1_checkbox = st.checkbox('Moving Average 1')
ma2_checkbox = st.checkbox('Moving Average 2')

fig7 = go.Figure()
fig7.add_trace(go.Scatter(x=tickerDf['Date'], y=tickerDf['Close'], name='Closing Price'))
 
if ma1_checkbox:
        days1 = st.slider('Business Days to roll MA1', 5, 200, 30)
        tickerDf['ma_1'] = tickerDf.Close.rolling(days1).mean()
        fig7.add_trace(go.Scatter(x=tickerDf['Date'], y=tickerDf['ma_1'], name='MA %s days'%days1))
if ma2_checkbox:
        days2 = st.slider('Business Days to roll MA2', 5, 200, 60)
        tickerDf['ma_2'] = tickerDf.Close.rolling(days2).mean()
        fig7.add_trace(go.Scatter(x=tickerDf['Date'], y=tickerDf['ma_2'], name='MA %s days'%days2))
      
fig7.update_layout(
    legend=dict(
        x=1,
        y=1,
        traceorder="normal",
        font=dict(
            family="sans-serif",
            size=11,
            color="black"
        ),
    )
)
fig7.update_layout(
    width=900,
    height=600,)
st.plotly_chart(fig7)

st.write("""
#### Volume by Date
""")

fig3 = px.line(tickerDf, x='Date', y='Volume')
fig3.update_layout(
    width=900,
    height=500,)

st.plotly_chart(fig3)



st.subheader('3. News Headlines')

parsed_news = scrape_news(ticker_name)
parsed_news = parsed_news[parsed_news['Title'].str.contains('Loading…',na=True)==False]
st.table(parsed_news.sort_values(by=['Date','Time'], ascending =False))

st.write(f"""
 *Source: Finviz*
""")


#import s&p500 stocks 
sp_500 = pd.read_csv(r'/Users/mirandacheng7/Desktop/Study/Harrisburg University/Research Methodology & Writing/Applied Project/data/S&P500.csv')

ticker_name =  st.session_state["my_input"]
company_name = sp_500[sp_500['Symbol'] == ticker_name].iloc[0]['Name']
sector_name = sp_500[sp_500['Symbol'] == ticker_name]['Sector']

today = date.today()
bday = today - BDay(4)
three_month = today - dateutil.relativedelta.relativedelta(months=3)
# dd/mm/YY
d1 = bday.strftime("%Y-%m-%d")
d2 = three_month.strftime("%Y-%m-%d")


st.write(f"""
# Sentiment Analysis
Here is the sentiment analysis for {ticker_name} - {company_name}.
""")

# model_name = st.sidebar.selectbox("Select a Sentiment Analysis Tool", ("TextBlob","NLTK"))

# let's apply the TextBlob API onto our tweet data to perform sentiment analysis!

parsed_news = scrape_news(ticker_name)
parsed_news = add_sentiment(parsed_news)

fig1 = px.histogram(parsed_news, x="polarity", nbins=20,title='Polarity Score')
fig1.update_layout(template = "simple_white")
fig1.update_layout(
    autosize=False,
    width=650,
    height=420,
    margin=dict(l=10,r=10,b=70,t=60,pad=1),)

st.plotly_chart(fig1)

fig2 = px.histogram(parsed_news, x="subjectivity", nbins=20,title='Subjectivity Score')
fig2.update_layout(template = "simple_white")
fig2.update_layout(
    autosize=False,
    width=650,
    height=420,
    margin=dict(l=10,r=10,b=70,t=60,pad=1),)

st.plotly_chart(fig2)

parsed_news['count'] =parsed_news.groupby('sentiment')['sentiment'].transform('count')
sentiment = parsed_news[['sentiment', 'count']].drop_duplicates()


fig3 = px.bar(sentiment, y='count', x='sentiment', text='count', color='sentiment')
fig3.update_traces(texttemplate = '%{text:.2s}',textposition='outside' )
# fig.write_image('bar.png')

st.plotly_chart(fig3)

fig4 = px.pie(sentiment, values='count', names='sentiment', title='Percentage of Polarity')
fig4.update_traces(textposition='inside',textfont_size = 16 )

st.plotly_chart(fig4)

sentiment_bydate = parsed_news.pivot_table(index= ['Date'], columns = ['sentiment'], aggfunc ='count').reset_index().iloc[:,:4]
sentiment_bydate.columns = ['date', 'Negative', 'Neutral','Positive']

column_names = ['Negative', 'Neutral','Positive']
sentiment_bydate['Total']= sentiment_bydate[column_names].sum(axis=1)

sentiment_bydate['positive percentage'] = sentiment_bydate['Positive']/sentiment_bydate['Total']
sentiment_bydate['negative percentage'] = sentiment_bydate['Negative']/sentiment_bydate['Total']
sentiment_bydate['neutral percentage'] = sentiment_bydate['Neutral']/sentiment_bydate['Total']



s_date = sentiment_bydate['date'].min()
start_date = datetime.strptime(s_date, '%m/%d/%Y')
start_date = start_date.strftime("%Y-%m-%d")


e_date = sentiment_bydate['date'].max()
end_date = datetime.strptime(e_date, '%m/%d/%Y')
end_date = end_date.strftime("%Y-%m-%d")
tickerData = yf.Ticker(ticker_name)
#get the historical prices for this ticker
tickerDf = tickerData.history(period='1d', start=start_date, end=end_date)
ticker_price = tickerDf.Close.reset_index()
ticker_price.columns = ['date', 'Close']
ticker_price['date'] = pd.to_datetime(ticker_price['date'])

ticker_price['date']  = ticker_price['date'].dt.strftime('%m/%d/%Y')



fig5 = px.bar(sentiment_bydate, x='date', y= sentiment_bydate.columns[1:4],title='Sentiment by Date')

# fig5.update_layout(template = "simple_white")
fig5.update_layout(
    autosize=False,
    width=900,
    height=500,
    margin=dict(l=10,r=10,b=70,t=60,pad=1),)
fig5.update_xaxes(tickangle=-50)
st.plotly_chart(fig5)


sentiment_bydate = sentiment_bydate.merge(ticker_price, how ='left', on ='date')
# st.table(ticker_price)

fig7 = make_subplots(specs=[[{"secondary_y": True}]])

fig7.add_trace(
    go.Scatter(x=sentiment_bydate['date'], y=sentiment_bydate['positive percentage'], name="Positive", mode="lines"),
    secondary_y=False
)

fig7.add_trace(
    go.Bar(x=sentiment_bydate['date'], y=sentiment_bydate['Close'], name="Stock Close Price"),
    secondary_y=True
)

fig7.update_xaxes(title_text="Stock Close Price")

# Set y-axes titles
fig7.update_yaxes(title_text="Positive Percentage", secondary_y=False)
fig7.update_yaxes(title_text="Stock Close Price", secondary_y=True)

st.plotly_chart(fig7)