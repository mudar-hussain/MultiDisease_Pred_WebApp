#!/usr/bin/env python
# coding: utf-8

# In[6]:


import numpy as np
import joblib
import streamlit as st


# In[7]:


ml_model = joblib.load(r'diabetes_ml_model')


# In[8]:


def diabetes_prediction(input):
  pred = ml_model.predict(input)
  if(pred[0] == 1):
    return 'Person is diabetic'
  else:
    return 'Person is not diabetic'


# In[9]:


def main():
    #Title
    st.title('Diabetes Prediction App')
    
    #getting input data from user
    Pregnancies = st.text_input('Number of Pregnancies')
    Glucose = st.text_input('Glucose Level')
    BloodPressure = st.text_input('Blood Pressure value')
    SkinThickness = st.text_input('Skin Thickness value')
    Insulin = st.text_input('Insulin Level')
    BMI = st.text_input('BMI value')
    DiabetesPedigreeFunction = st.text_input('Diabetes Pedigree Function value')
    Age = st.text_input('Age of the Person')
    
    #code for prediction
    diagnosis = ''
    
    #creating a button for prediction
    if st.button('Diabetes Test Result'):
        input = [[Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]]
        diagnosis = diabetes_prediction(input)
        st.success(diagnosis)
    
    


# In[5]:


if __name__ == '__main__':
    main()
