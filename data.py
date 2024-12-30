import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
import tensorflow as tf
start='2012-01-01'
end='2022-12-21'
stock='GOOG'
data=yf.download(stock,start,end)
#firstly reset the dates to indices so that for ore clearity
data.reset_index(inplace=True)
#calculate the moving average to depict the trends in stock market predictor
ma_100_days=data.Close.rolling(100).mean()
plt.figure(figsize=(8,6))
plt.plot(ma_100_days,'r')
plt.plot(data.Close,'g')
ma_200_days=data.Close.rolling(200).mean()
plt.figure(figsize=(8,6))
plt.plot(ma_100_days,'r')
plt.plot(ma_200_days,'b')
plt.plot(data.Close,'g')

data.dropna(inplace=True)
#train test spliting of data
data_train=pd.DataFrame(data.Close[0:int(len(data)*0.80)])
data_test=pd.DataFrame(data.Close[int(len(data)*0.80):len(data)])
#now we need to do scalling of the dataset to convert it into a range between ones
from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler(feature_range=(0,1))
data_train_scale=scaler.fit_transform(data_train)
x=[]
y=[]
for i in range(100,data_train_scale.shape[0]):
    x.append(data_train_scale[i-100:i])
    y.append(data_train_scale[i,0])
x,y=np.array(x),np.array(y)  
    
#Model Creation
from keras.layers import Dense,Dropout,LSTM
from keras.models import Sequential
model=Sequential()
model.add(LSTM(units=50,activation='relu',return_sequences=True
               ,input_shape=(x.shape[1], 1)
))
model.add(Dropout(0.2))

model.add(LSTM(units=60,activation='relu',return_sequences=True))
model.add(Dropout(0.3))

model.add(LSTM(units=80,activation='relu',return_sequences=True))
model.add(Dropout(0.4))

model.add(LSTM(units=120,activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(units=1))

model.compile(optimizer='adam',loss='mean_squared_error')
#fitting of the model 
model.fit(x,y,epochs=50,batch_size=32,verbose=1)
model.summary()

#Test the data
past_100_days=data_train.tail(100)
data_test=pd.concat([past_100_days,data_test],ignore_index=True)
data_test_scale=scaler.fit_transform(data_test)
#use the slicing again
x=[]
y=[]
for i in range(100,data_test_scale.shape[0]):
    x.append(data_test_scale[i-100:i])
    y.append(data_test_scale[i,0])
  
x,y=np.array(x),np.array(y)  
y_predict=model.predict(x)

#Convertiing into again the scaler
scale=1/scaler.scale_

y_predict=y_predict*scale

y=y*scale
plt.figure(figsize=(10,8))
plt.plot(y_predict,'r',label='Predicted Price')
plt.plot(y,'g',label="Atcual Price")
plt.xlabe("Time")
plt.ylabel("Price")
plt.legend()
plt.show()
#Saving the model
model.save("Stock market Prediction using keras")

    