import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
from pandas_datareader import data as pdr
import yfinance as yf
from keras.models import load_model
import streamlit as st
from datetime import date, datetime
import plotly.express as px
import pandas_market_calendars as mcal
import dateutil.relativedelta
from plotly import graph_objs as go
from sklearn.preprocessing import MinMaxScaler
from PIL import Image


image = Image.open('images/working.jpg')
st.sidebar.image(image, caption='',
                 width=400, use_column_width='auto')

ticker_name =  st.session_state["my_input"]

#import s&p500 stocks 
sp_500 = pd.read_csv(r'/Users/mirandacheng7/Desktop/Study/Harrisburg University/Research Methodology & Writing/Applied Project/data/S&P500.csv')
company_name = sp_500[sp_500['Symbol'] == ticker_name].iloc[0]['Name']

st.write(f"""
# Stock Price Prediction 💎
##### Company: {company_name}
This page allows you to view stock price predictions from LSTM model.
###### Disclaimer:  Nothing in the page constitutes professional and/or financial advice.
""")

start = st.date_input('Start',value=pd.to_datetime('2010-01-01'))
end = st.date_input('End', value = pd.to_datetime('today'))
st.write('How many days do you want to base your predictions on?')
steps = st.selectbox(label='Pick a prediction day', options=[30,60,100],index=1)


yf.pdr_override()
df = pdr.get_data_yahoo(ticker_name, start, end)
df = df.reset_index()

#Describe data
st.subheader('1.Data Loading 🏋️')
st.markdown(f"""<p class='a'>Stock Price from {start} to {end}</p>""", unsafe_allow_html=True)
st.write("""##### Raw Data""")

st.write(df.head())

#split the data into trainning and testing
data_training = pd.DataFrame(df['Close'][0:int(len(df)*0.8)])
data_testing = pd.DataFrame(df['Close'][int(len(df)*0.8):int(len(df))])


scaler = MinMaxScaler(feature_range=(0,1))
#predict the closing price
#scale down the trainning data and transform the data into an array
data_training_array = scaler.fit_transform(data_training)

# divid the data into x_train and y_train
x_train = []
y_train = []

#the value of the 101th day(y_train) will be depend on the values from the previous 100 days (x_train)
for i in range(steps, data_training_array.shape[0]):
    x_train.append(data_training_array[i-steps:i])
    y_train.append(data_training_array[i,0])
    
x_train, y_train = np.array(x_train),np.array(y_train)

#load Model
model = load_model('keras_model.h5')
model_30 = load_model('keras_model_30.h5')
model_60 = load_model('keras_model_60.h5')

past_100_days = data_training.tail(steps)
final_df = past_100_days.append(data_testing,ignore_index=True)
input_data = scaler.fit_transform(final_df)
x_test = []
y_test =[]
for i in range(steps, input_data.shape[0]):
    x_test.append(input_data[i-steps: i])
    y_test.append(input_data[i,0])
x_test, y_test = np.array(x_test),np.array(y_test)

x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1],1))

if steps == 100:
    y_predicted = model.predict(x_test)
elif steps == 60:
    y_predicted = model_60.predict(x_test)
elif steps == 30:
    y_predicted = model_30.predict(x_test)
    

y_test = scaler.inverse_transform(y_test.reshape(-1,1))
y_predicted = scaler.inverse_transform(y_predicted.reshape(-1,1))
data = pd.DataFrame(df['Date'][int(len(df)*0.8):int(len(df))])
data['Actuals'] = y_test
data['Forecasts(LSTM)'] = y_predicted

st.subheader('2.Stock Price Prediction Using LSTM Models')
st.write('The predictions are based on %s days in the past'%steps)
# Plot 
df_test = df.iloc[0:int(len(df)*0.8),:]

fig2 = go.Figure()
fig2.add_trace(go.Scatter(x=df_test['Date'], y=df_test['Close'], name='Actual Stock Price (Train)'))
fig2.add_trace(go.Scatter(x=data['Date'], y=data['Actuals'], name='Actual Stock Price (Valid)'))
fig2.add_trace(go.Scatter(x=data['Date'], y=data['Forecasts(LSTM)'], name='Forecasted Stock Price (LSTM %s days)'%steps))

fig2.update_layout(
    width=900,
    height=500,)

st.plotly_chart(fig2)

st.subheader('3.Actual Stock Price vs. Predicted Stock Price Using LSTM')
# Plot 
fig3 = go.Figure()
fig3.add_trace(go.Scatter(x=data['Date'], y=data['Actuals'], name='Actual Stock Price'))
fig3.add_trace(go.Scatter(x=data['Date'], y=data['Forecasts(LSTM)'], name='Forecasted Stock Price(LSTM%s days)'%steps))
fig3.update_layout(
    width=900,
    height=500,)

st.plotly_chart(fig3)

rmse = np.sqrt(np.mean(y_predicted - y_test)**2)
st.write(f"""##### Square Root Error: {rmse}""")


# y_predicted_60 = model_60.predict(x_test)
# y_predicted_60 = scaler.inverse_transform(y_predicted_60.reshape(-1,1))
# y_predicted_30 = model_30.predict(x_test)
# y_predicted_30 = scaler.inverse_transform(y_predicted_30.reshape(-1,1))

# st.subheader('Compare Predictions Based on Different Days of Historical Prices')
# # # Plot 
# # fig3 = go.Figure()
# # fig3.add_trace(go.Scatter(x=data['Date'], y=data['Actuals'], name='Actual Stock Price'))
# # fig3.add_trace(go.Scatter(x=data['Date'], y=data['Forecasts(LSTM)'], name='Forecasted Stock Price(LSTM%s days)'%steps))
# # fig3.update_layout(
# #     width=900,
# #     height=500,)

# # st.plotly_chart(fig3)
