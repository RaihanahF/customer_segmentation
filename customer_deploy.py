# -*- coding: utf-8 -*-
"""
Created on Wed May 18 16:35:58 2022

@author: Fatin
"""

import pickle
import os
import pandas as pd
from tensorflow.keras.models import load_model
import numpy as np
from customer_modules import ExploratoryDataAnalysis

OHE_SCALER_PATH = os.path.join(os.getcwd(), 'saved_model', 'ohe_scaler.pkl')
SS_SCALER_PATH = os.path.join(os.getcwd(), 'saved_model', 'ss_scaler.pkl')
MODEL_PATH = os.path.join(os.getcwd(), 'saved_model', 'model.h5')
DATA_PATH = os.path.join(os.getcwd(),'new_customers.csv')

ohe_scaler = pickle.load(open(OHE_SCALER_PATH, 'rb'))
ss_scaler = pickle.load(open(SS_SCALER_PATH, 'rb'))

model = load_model(MODEL_PATH)
model.summary()

customer_segmentation = {0:'A', 1:'B', 2:'C', 3:'D', 4:'E', 5:'F'}

df_new = pd.read_csv(DATA_PATH)

eda = ExploratoryDataAnalysis()

# Label Encoder for Columns with Non-integer dtype
df_new = eda.label_encoder(df_new)


# Fill NaNs using Iterative Imputer
df_new = eda.iterative_imputer(df_new.iloc[:,:-1])
df_new = df_new.round(0).astype(int)

segmentation = []

new_customers = df_new.values

for i in range (len(new_customers)):

    new_customers_scaled = ss_scaler.transform(np.expand_dims((new_customers[i]), axis=0))
    outcome = model.predict(new_customers_scaled)
    segmentation.append(customer_segmentation[np.argmax(outcome)])


df_updated = pd.DataFrame(new_customers,
                          columns = df_new.columns)

df_updated['Segmentation'] = segmentation

df_updated.to_csv("new_customers_updated.csv")
