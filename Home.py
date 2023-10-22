import streamlit as st
import pandas as pd
import yfinance as yf
from PIL import Image

st.set_page_config(
    page_title = "Stock Analysis Web Application"
)

image = Image.open('images/bull.jpg')



st.sidebar.image(image, caption='',

                 use_column_width=True)
        

st.sidebar.header('A stock analysis web app')
        
st.title("Stock Analysis Web App 💸")
st.write(f"""
## About this app
This app allows you to analyze any stocks in S&P 500. You can view the stock price, read latest news about your investment, and apply state-of-art machine learning techniques to predict future stock prices. 
\n **Have fun!** \n
""")



st.write(f"""
### S&P 500 Stocks
""")

sp_500 = pd.read_csv(r'/Users/mirandacheng7/Desktop/study/Harrisburg University/Research Methodology & Writing/Applied Project/data/S&P500.csv')
st.sidebar.success("Select a page above.")

ticker = st.selectbox('Select the ticker if present in the S&P 500 index', sp_500['Symbol'], index = 30).upper()
pivot_sector = True
checkbox_noSP = st.checkbox('Select this box to write the ticker (if not present in the S&P 500 list). \
                            Deselect to come back to the S&P 500 index stock list')
if checkbox_noSP:
    ticker = st.text_input('Write the ticker (check it in yahoo finance)', 'MN.MI').upper()
    pivot_sector = False
    
st.session_state["my_input"] = ticker
# if "my_input" not in st.session_state:
#     st.session_state["my_input"] = ""

# my_input = st.text_input("Pick a stock to start your analysis", st.session_state["my_input"])
# submit = st.button("Submit")

# if submit:
#     st.session_state["my_input"] = my_input
#     st.write("You have chosed stock", my_input)


st.table(sp_500)

