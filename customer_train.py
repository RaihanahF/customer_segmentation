# -*- coding: utf-8 -*-
"""
Created on Wed May 18 17:18:05 2022

@author: Fatin
"""

import os
import pandas as pd
import pickle
import numpy as np
from customer_modules import ExploratoryDataAnalysis, ModelCreation, ModelEvaluation
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from tensorflow.keras.utils import plot_model

MODEL_SAVE_PATH = os.path.join(os.getcwd(),'model.h5')
SS_SCALER_PATH = os.path.join(os.getcwd(), 'saved_model', 'ss_scaler.pkl')
OHE_SCALER_PATH = os.path.join(os.getcwd(), 'saved_model', 'ohe_scaler.pkl')
LOG_PATH = os.path.join(os.getcwd(),'log')

# EDA

# Step 1: Data loading
TRAIN_DATA_PATH = os.path.join(os.getcwd(),'train.csv')
df = pd.read_csv(TRAIN_DATA_PATH)

# Step 2: Data inspection/ visualization
df.info()
df.describe()

print(df.nunique())
print(df.isna().sum())

bool_series = pd.DataFrame(df).duplicated()
sum(bool_series==True)

# Step 3: Data cleaning

eda = ExploratoryDataAnalysis()

# Label Encoder for Columns with Non-integer dtype
dummy_df = eda.label_encoder(df)

dummy_df.info()

# Fill NaNs using Iterative Imputer
dummy_df = eda.iterative_imputer(dummy_df)
dummy_df = dummy_df.round(0).astype(int)

# Step 4: Features selection
# Step 5:Data preprocessing

X = dummy_df.drop(['Segmentation'], axis = 1).values
y = dummy_df['Segmentation']

X_train, X_test, y_train, y_test  = train_test_split(X, y, test_size=0.3, random_state=123)
 
# Pipelne creation
# 1) Min-max scaler OR Standard Scaler
# 2) Linear Regression
eda.pipeline_scaler(X_train,y_train,X_test,y_test)

# Standard Scaler

ss_scaler = StandardScaler()
X_scaled = ss_scaler.fit_transform(X)
pickle.dump(ss_scaler, open(SS_SCALER_PATH,'wb'))

ohe_scaler = OneHotEncoder(sparse=False)
y_one_hot = ohe_scaler.fit_transform(np.expand_dims(y, axis=-1))
pickle.dump(ohe_scaler, open(OHE_SCALER_PATH, 'wb'))

X_train, X_test, y_train, y_test  = train_test_split(X_scaled, y_one_hot, test_size=0.3, random_state=123)


#%% Model Creation

# Functional API

# Step 1: Model creation
mc = ModelCreation()
model = mc.create_model(4, (10,), nb_nodes=128, activation='relu', dropout=0.3)

plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)

# Step 2: wrap your sequential api
model.compile(optimizer='adam', 
              loss='categorical_crossentropy',
              metrics='acc')

# Step 4) Callbacks
import datetime
log_files = os.path.join(LOG_PATH, 
                         datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))

from tensorflow.keras.callbacks import TensorBoard
tensorboard_callback = TensorBoard(log_dir=log_files, histogram_freq=1)

from tensorflow.keras.callbacks import EarlyStopping
early_stopping_callback = EarlyStopping(monitor='val_loss', patience=3)

# Step 5) Model training
hist = model.fit(X_train,y_train, epochs=50, validation_data=(X_test,y_test),
                 callbacks=[tensorboard_callback, early_stopping_callback])

# Save model
model.save(MODEL_SAVE_PATH)

me = ModelEvaluation()

# Visualize the training process using matplotlib, if tensorboard not working
me.training_history(hist)

# Report generation (F-1 score, precision, acc.)
me.report_generation(X_test, y_test, model, 4)
