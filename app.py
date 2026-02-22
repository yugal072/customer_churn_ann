import os
os.environ['TF_USE_LEGACY_KERAS'] = '1'  # ← This is all you need!

import streamlit as st
import tensorflow as tf
import numpy as np 
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder 
import pandas as pd 
import pickle
from tensorflow.keras.layers import InputLayer

# Quick monkey-patch to handle batch_shape → batch_input_shape rename
original_init = InputLayer.__init__

def patched_init(self, *args, **kwargs):
    if 'batch_shape' in kwargs:
        # Convert batch_shape to what older InputLayer expects
        batch_shape = kwargs.pop('batch_shape')
        if batch_shape and batch_shape[0] is None:
            # batch_shape = [None, ...] → batch_input_shape = [None, ...]
            kwargs['batch_input_shape'] = batch_shape
        else:
            # Rare case, fallback to input_shape if no batch dim
            kwargs['input_shape'] = batch_shape[1:] if len(batch_shape) > 1 else ()
    return original_init(self, *args, **kwargs)

InputLayer.__init__ = patched_init

# Load trained model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model('model.h5')

model = load_model()           # now only runs when needed
with open('label_encoder_gender.pkl','rb') as file:
    label_encoder_gender = pickle.load(file)

with open('onehot_encoder_geo.pkl','rb') as file:
    onehot_encoder_geo = pickle.load(file)

with open('scaler.pkl','rb') as file:
    scaler = pickle.load(file)

# Streamlit app
st.title("Customer churn predictions") 

# User Input
geography = st.selectbox('Geography', onehot_encoder_geo.categories_[0])
gender = st.selectbox('Gender', label_encoder_gender.classes_)
age = st.slider('Age', 18,92)
balance = st.number_input('Balance')
credit_score = st.number_input('Credit Score')
estimated_salary = st.number_input('Estimated Salary')
tenure = st.slider('Tenure', 0,10)
num_of_products = st.slider('Number of Products', 1,4)
has_cr_card = st.selectbox('Has Credit Card', [0,1])
is_active_member = st.selectbox('Is Active Member', [0,1])

# Prepare the input data
input_data= pd.DataFrame({
    'CreditScore': [credit_score],
    'Gender': [label_encoder_gender.transform([gender])[0]],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member],
    'EstimatedSalary': [estimated_salary]
})

# One hot encode geography
geo_encoded = onehot_encoder_geo.transform([[geography]])
geo_encoded_df = pd.DataFrame(geo_encoded, columns = onehot_encoder_geo.get_feature_names_out(['Geography']))

# Combine one hot encoded columns with input data
input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)

# Scale the input data
input_data_scaled = scaler.transform(input_data)

# Predict churn
prediction = model.predict(input_data_scaled)
prediction_prob = prediction[0][0]

# Display result
if prediction_prob >= 0.5:
    st.write("Customer will churn")
else:
    st.write("Customer will not churn")



