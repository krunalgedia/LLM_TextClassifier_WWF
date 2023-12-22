#!/usr/bin/env python
# coding: utf-8

import os
import pandas as pd
import numpy as np

class presticker_compute():
    def __init__(self, presticker, version, df:None, df2:None, label_map:None, question:None):
        assert type(presticker)== str, f"string presticker required, input type: {type(presticker)}"
        assert version in ['v1','v2'], f"Version should be either v1 or v2"
        self.presticker = presticker
        self.version = version
        self.df = df
        self.df2 = df2
        self.label_map = label_map
        self.question = question
        
    def get_presticker(self):
        if self.version=='v1':
            return self.presticker
        if self.version=='v2':
            assert type(self.df) ==pd.core.frame.DataFrame, f"Pandas df required, input dtype: {type(self.df)}"
            assert type(self.df2) ==pd.core.frame.DataFrame, f"Pandas df required, input dtype: {type(self.df2)}"
            assert type(self.label_map) == dict, f"Dict required, input dtype: {type(self.label_map)}"
            assert type(self.question) == str, f"str required, input dtype: {type(self.question)}"
            self.presticker = self.prestick_keypoints(self.df2, self.presticker)
            self.presticker = self.prestick_reason(self.df, self.presticker, self.label_map)
            self.presticker = self.prestick_question(self.presticker, self.question)
            return self.presticker 
             
    def prestick_keypoints(self, df, presticker):
        for col in df.columns:
            self.presticker += col
            self.presticker += '\n'
            self.presticker += df.loc[:,col].str.cat(sep='\n')
            self.presticker += '\n'
        return self.presticker   

    def prestick_reason(self, df, presticker, label_map):
        for k,v in label_map.items():
            self.presticker += v
            self.presticker += "\n"
            self.presticker += df.loc[df['label']==k,'reason'].str.cat(sep='\n')
            self.presticker += "\n"
        return self.presticker    

    def prestick_question(self, presticker, question):
        self.presticker += "\n"
        self.presticker += question
        self.presticker += "\n"
        return self.presticker            

class poststicker_compute():
    def __init__(self, poststicker, version:None):
        assert type(poststicker)==str, f"string poststicker required, input type: {type(poststicker)}"
        assert version in ['v1','v2'], f"Version should be either v1 or v2"
        self.version = version
        self.poststicker = poststicker
     
    def get_poststicker(self):
        if self.version=='v1':
            return self.poststicker
        if self.version=='v2':
            self.poststicker = ''
            return self.poststicker
