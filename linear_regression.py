# linear regression for advertisemet sales prediction 
import numpy as np
import pandas as pd
import pickle
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,r2_score
import matplotlib.pyplot as plt

df = pd.read_csv("C:\\Users\\acer\\Downloads\\advertising_dataset.csv")
print(df.head())
X_train, X_test, y_train, y_test = train_test_split(df.iloc[:,:-1],df.iloc[:,-1], test_size=0.2, random_state=42)
# print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
lr = LinearRegression()
lr.fit(X_train,y_train)
y_pedict=lr.predict(X_test)
# print(y_pedict)
mean_err= mean_squared_error(y_test,y_pedict)
# print(mean_err)
r_score= r2_score(y_test,y_pedict)
# print(r_score)
tv_budget=219.6
radio_budget=15.5
newspaper_budget=69.8
def predict_sales(tv_budget, radio_budget, newspaper_budget):
    features = np.array([[tv_budget, radio_budget, newspaper_budget]])
    result = lr.predict(features).reshape(1, -1)
    return result[0]
sales = predict_sales(tv_budget, radio_budget, newspaper_budget)
# print(sales)
pickle.dump( lr, open( "model.pkl", "wb" ) )

model = pickle.load(open( "model.pkl", "rb" ) )
st.title("Advertising Sales Prediction")
TV = st.text_input("Enter TV Budget")
Radio = st.text_input("Enter Radio Budget")
Newspaper= st.text_input("Enter Newspaper Budget")

if st.button("Predict"):
    features = np.array([[float(TV), float(Radio), float(Newspaper)]])
    result= model.predict(features).reshape(1,-1)
    st.write("Predicted Sales: ",result[0])


