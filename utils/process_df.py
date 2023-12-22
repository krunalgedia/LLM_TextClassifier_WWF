#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import random
from sklearn.model_selection import train_test_split


class process_df():
    def __init__(self, df):
        assert type(df) ==pd.core.frame.DataFrame, f"Pandas df required, input dtype: {type(df)}"
        self.df = df
        
    def skim_cols(self, 
                  df, 
                  keep_cols=['Inhalt','Konstruktiv (1= eher konstruktiv   0 = eher nicht konstruktiv '], 
                  renamed_cols = ['content','label']):
        self.df = df.drop(columns=[i for i in list(df.columns) if i not in keep_cols])
        self.df.columns = renamed_cols
        return self.df
    
    def clean_df(self, df): 
        df.dropna(inplace=True)
        #df.label = df.label.map({'constructive':1,'not constructive':0})
        #df.dropna(inplace=True)
        df.reset_index(drop=True)
        self.df = df.loc[(df.label==1)|(df.label==0)]
        return self.df
    
    def process_and_split_df(self, df):    
        self.df = df.astype({'label':int})
        Xy_train, Xy_test, y_train, y_test = train_test_split(self.df, self.df.label, test_size=0.2, stratify=self.df.label, random_state=42)
        Xy_train.reset_index(inplace=True, drop=True)
        Xy_test.reset_index(inplace=True, drop=True)
        Xy_train_series = Xy_train.apply(lambda row: f"Text: {row[Xy_train.columns[0]]} \n Class:{row[Xy_train.columns[1]]} \n \n", axis=1)
        return Xy_train_series,Xy_test,Xy_train,Xy_test 
