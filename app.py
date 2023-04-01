import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf
import streamlit as st
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import  r2_score


st.title('Stock Trend Predection ')
st.subheader('On ML Model')

user_input = st.text_input('Enter Stock Ticker      eg.....TSLA, AAPL,TATAMOTORS.NS,etc', 'AAPL')
df = yf.Ticker(user_input)
period = st.selectbox('Select Years or Time',['10Y','MAX'])
df1 = pd.DataFrame(df.history(period=period))

## Describing data
st.subheader(f'Data for previous {period}')
st.write(df1.head(5))

##Visualisation
st.subheader('Closing Price Vs Time Chart')
fig = plt.figure(figsize=(12,6))
plt.plot(df1.Close)
st.pyplot(fig)

ma100 = df1.Close.rolling(100).mean()
ma200 = df1.Close.rolling(200).mean()
st.subheader('Closing Price Vs Time Chart with 100 Days Moving Avg')
fig = plt.figure(figsize=(12,6))
plt.plot(df1.Close)
plt.plot(ma100)
st.pyplot(fig)

st.subheader('Closing Price Vs Time Chart with 100 and 200 Days Moving Avg')
fig = plt.figure(figsize=(12,6))
plt.plot(df1.Close)
plt.plot(ma100)
plt.plot(ma200)
st.pyplot(fig)


##Load model

model = load_model('keras_model.h5')

#Training data 
data_training =pd.DataFrame(df1['Close'][0:int(len(df1)*0.70)])
data_testing = pd.DataFrame(df1['Close'][int(len(df1)*0.70):int(len(df1))]) 


scaler = MinMaxScaler(feature_range=(0,1))

past_100_days = data_training.tail(100)
final_df = pd.concat([past_100_days,data_testing], ignore_index =  True)
input_data = scaler.fit_transform(final_df)


x_test =[]
y_test = []

for i in range(100,input_data.shape[0]):
    x_test.append(input_data[i-100:i])
    y_test.append(input_data[i,0])
x_test , y_test =np.array(x_test), np.array(y_test)
y_predected = model.predict(x_test)

scale_factor = scaler.scale_[0]
y_predected = y_predected * scale_factor
y_test = y_test * scale_factor


##Final Graph
st.subheader('Predictions Vs Original')
fig2 = plt.figure(figsize=(12,6))
plt.plot(y_test,'b',label = 'Original Price')
plt.plot(y_predected, "r", label ='Predicted Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig2)

Accuracy =r2_score(y_true=y_test,y_pred=y_predected) 

st.success(f'Accuracy of Model : {np.round((Accuracy * 100),2)}%')