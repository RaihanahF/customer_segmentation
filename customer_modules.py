# -*- coding: utf-8 -*-
"""
Created on Wed May 18 17:29:27 2022

@author: Fatin
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import Lasso
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.utils import plot_model
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import ConfusionMatrixDisplay
import os

MODEL_SAVE_PATH = os.path.join(os.getcwd(),'model.h5')
SS_SCALER_PATH = os.path.join(os.getcwd(), 'saved_model', 'ss_scaler.pkl')
OHE_SCALER_PATH = os.path.join(os.getcwd(), 'saved_model', 'ohe_scaler.pkl')
LOG_PATH = os.path.join(os.getcwd(),'log')


class ExploratoryDataAnalysis():
    
    def __init__(self):
        
        pass
    
    def label_encoder(self, df):
        '''
        This function will encode the input data using label encoder approach.
    
        Parameters
        ----------
        df : DataFrame
            df will undergo label encoding.
    
        Returns
        -------
        Label-encoded df in DataFrame format
    
        '''
        label_encoder = LabelEncoder()
    
        # Get  columns whose data type is object
        filtered_cols = df.dtypes[df.dtypes == np.object]
        filtered_cols_list = list(filtered_cols.index)
        
        for col in filtered_cols_list:
            df[col] = label_encoder.fit_transform(df[col])
        
        return df
    
    def iterative_imputer(self, df):
        
        imputer = IterativeImputer(max_iter=10, random_state=0)
        imputer.fit(df)
        updated_df = imputer.transform(df)
        updated_df = pd.DataFrame(updated_df)
        updated_df.columns = df.columns
        
        return updated_df
       
    def feature_selection(self, x, y):
        '''
        This function plot lasso coefficient to find top features.
    
        Parameters
        ----------
        x_train : Array
            x contains the features.
        y_train : Array
            y contains the label.
    
        Returns
        -------
        A graph plotted using matplotlib.
    
        '''
        lasso = Lasso()
        lasso.fit(x,y)
        lasso_coef = lasso.coef_ # to obtain coefficients
        print(lasso_coef) # select non-zero coefficiens
        
        # graphs
        plt.figure(figsize=(10,10))
        plt.plot(x.columns,abs(lasso_coef))
        plt.grid()
            
        x = x.iloc[:,lasso_coef!=0]
        
        return x
    
    def pipeline_scaler(self, X_train,y_train,X_test,y_test):
        '''
        This function perform comparison of Standard Scaler and MinMax Scaler
        with Logistic Regression to find the more effective scaler.
    
        Parameters
        ----------
        X_train : Array
            DESCRIPTION.
        y_train : Array
            DESCRIPTION.
        X_test : Array
            DESCRIPTION.
        y_test : Array
            DESCRIPTION.
    
        Returns
        -------
        None.
    
        '''
        
        steps_ss = [('Standard Scaler', StandardScaler()), 
                ('Logistic Regression', LogisticRegression())
        ]
        # Min-Max Scaler Steps
        steps_mm = [('Min Max Scaler', MinMaxScaler()), 
                    ('Logistic Regression', LogisticRegression())
        ]
        pipeline_ss = Pipeline(steps_ss) # To load the steps into the pipeline
        pipeline_mm = Pipeline(steps_mm) # To load the steps into the pipeline
        pipelines = [pipeline_ss, pipeline_mm] # create a list to store all pipelines
        
        for pipe in pipelines:
            pipe.fit(X_train,y_train)
        
            print(str(pipe))
            print('Training set score: ' + str(pipe.score(X_train,y_train)))
            print('Test set score: ' + str(pipe.score(X_test,y_test)) + '\n')

class ModelCreation():
    
    def __init__(self):
        pass
    
    def create_model(self, nb_classes, input_data_shape, nb_nodes=32, activation='relu', dropout=0.2):
        '''
        This function creates a model with 2 hidden layers. With
        last layer of the model comprises of softmax activation function
    
        Parameters
        ----------
        nb_class : Int
            Contains the number of classes for output layer.
        input_data_shape : Array
            Contains the shape of the model.
        nb_nodes : Int, optional
            Contains number of nodes for hidden layer. The default is 32.
        activation : String, optional
            Contains the type of activation for layers. The default is 'relu'.
        dropout : Int, optional
            Contains the dropout rates. The default is 0.2.
    
        Returns
        -------
        model : tensor
            A created model.
    
        '''
        input_1 = Input(shape=input_data_shape)    
        hidden_1 = Dense(nb_nodes, activation=activation)(input_1)
        bn_1 = BatchNormalization()(hidden_1)
        drop_layer_1 = Dropout(dropout)(bn_1)
        hidden_2 = Dense(nb_nodes, activation=activation)(drop_layer_1)
        bn_2 = BatchNormalization()(hidden_2)
        drop_layer_2 = Dropout(dropout)(bn_2)
        hidden_3 = Dense(nb_nodes, activation=activation)(drop_layer_2)
        bn_3 = BatchNormalization()(hidden_3)
        #drop_layer_3 = Dropout(dropout)(bn_3)
        output_1 = Dense(nb_classes, activation='softmax')(bn_3)
    
        model = Model(inputs=[input_1], outputs=[output_1])
        model.summary()
    
        plot_model(model)
        
        return model

class ModelEvaluation():
    
    def __init__(self):
        pass
    
    def training_history(self, hist):
        '''
        This function plot the loss and metric in training and validation using matplotlib.
    
        Parameters
        ----------
        hist : callbacks
            History callback created after model training.
    
        Returns
        -------
        None.
    
        '''
        keys = [i for i in hist.history.keys()]
        training_loss = hist.history[keys[0]]
        training_metric = hist.history[keys[1]]
        validation_loss = hist.history[keys[2]]
        validation_metric = hist.history[keys[3]]
    
        # Plot loss
        plt.figure()
        plt.plot(training_loss) # during training
        plt.plot(validation_loss) # validation loss
        plt.title('training {} and validation {}'.format(keys[0], keys[0]))
        plt.xlabel('epoch')
        plt.ylabel(keys[0])
        plt.legend(['training loss', 'validation loss'])
        plt.show()
        
        # Plot metric
        plt.figure()
        plt.plot(training_metric) # during training
        plt.plot(validation_metric) # validation loss
        plt.title('training {} and validation {}'.format(keys[1], keys[1]))
        plt.xlabel('epoch')
        plt.ylabel(keys[1])
        plt.legend(['training acc', 'validation acc'])
        plt.show()
        

    def report_generation(self, x_test, y_test, model, label):
        '''
        This function generate Confusion Matrix and classification report to show the precision of the model. 
    
        Parameters
        ----------
        x_test : Array
            Contains the features being tested.
        y_test : Array
            Contains the target value to be predicted.
    
        Returns
        -------
        None.
    
        '''
    
        pred_x = model.predict(x_test)
        y_true = np.argmax(y_test, axis=1)
        y_pred = np.argmax(pred_x, axis=1)
        cm = confusion_matrix(y_true, y_pred)
        cr = classification_report(y_true, y_pred)
        print(cr)
    
        labels = [str(i) for i in range(label)]    
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
        disp.plot(cmap=plt.cm.Blues)
        plt.show()