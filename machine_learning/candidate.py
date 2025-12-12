import pandas as pd
import numpy as np

data = pd.DataFrame([
    ['Sunny', 'Warm', 'Normal', 'Strong', 'Warm', 'Same',           'Yes'],
    ['Sunny', 'Warm', 'High',   'Strong', 'Warm', 'Same',           'Yes'],
    ['Rainy', 'Cold',  'High',   'Strong', 'Warm', 'Change',         'No'],
    ['Sunny', 'Warm',  'High',   'Strong', 'Cool', 'Change',         'Yes']
], columns=['Sky','AirTemp','Humidity','Wind','Water','Forecast','PlayTennis    '])

def candidate_elimination(data):
    AttributeErrors = data.columns[:-1]
    S = ['0'] * len(AttributeErrors)
    g =[['?'] * len(attribute)]

    for index, row in data.iterrows():
        x = row[-1]
        y= row[-1]

        if y == 'yes':
         
            G = [g for g in G if all(g[i]=='?' or g[i]==x[i] for i in range(len(attributes)))]
           
            for i in range(len(attributes)):
                if S[i] == '0':
                    S[i] = x[i]
                elif S[i] != x[i]:
                    S[i] = '?'

                else:

            G_new = []
            for g in G:
                if all (g[i]=='?' or g[i]!=x[i]) for i in range (len)