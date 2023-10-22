import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.express as px
from PIL import Image

st.write("""
         # Stock Performance Comparison 📊
         This page allows you to compare stock prices and returns of multiple stocks""")
df_ticker = pd.read_csv(r'/Users/mirandacheng7/Desktop/Study/Harrisburg University/Research Methodology & Writing/Applied Project/data/S&P500.csv')
tickers = df_ticker['Symbol']
ticker_name = st.session_state["my_input"]
dropdown = st.multiselect('Pick your stocks', tickers,ticker_name)

start = st.date_input('Start',value=pd.to_datetime('2022-01-01'))
end = st.date_input('End', value = pd.to_datetime('today'))
image = Image.open('images/image4.jpg')
st.sidebar.image(image, caption='',
                 use_column_width=True)

company_list = df_ticker[df_ticker['Symbol'].isin(dropdown)]['Name'].to_list()

@st.cache
def relativeret(df):
    rel = df.pct_change()
    cumret = (1+rel).cumprod() -1
    cumret = cumret.fillna(0)
    return cumret
if len(dropdown) > 0:
    # @st.cache
    df = yf.download(dropdown, start, end)['Close']
    st.subheader('Stock Prices of '+', '.join(company_list) )
    st.line_chart(df)
    st.subheader('Return of '+', '.join(company_list))
    df_return = relativeret(df)
    st.line_chart(df_return)