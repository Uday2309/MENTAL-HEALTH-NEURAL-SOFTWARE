import pandas as pd


data = pd.DataFrame([
    ['Sunny', 'Warm', 'Normal', 'Strong', 'Warm', 'Same', 'Yes'],
    ['Sunny', 'Warm', 'High', 'Strong', 'Warm', 'Same', 'Yes'],
    ['Rainy', 'Cold', 'High', 'Strong', 'Warm', 'Change', 'No'],
    ['Sunny', 'Warm', 'High', 'Strong', 'Cool', 'Change', 'Yes']
], columns=['Sky','AirTemp','Humidity','Wind','Water','Forecast','PlayTennis'])


def candidate_elimination(data):
    attributes = data.columns[:-1]
    S = ['0'] * len(attributes)
    G = [['?'] * len(attributes)]
    
    for index, row in data.iterrows():
        x = row[:-1].values
        y = row[-1]
        
        if y == 'Yes':
         
            G = [g for g in G if all(g[i]=='?' or g[i]==x[i] for i in range(len(attributes)))]
           
            for i in range(len(attributes)):
                if S[i] == '0':
                    S[i] = x[i]
                elif S[i] != x[i]:
                    S[i] = '?'
        else:
            
            G_new = []
            for g in G:
                if all(g[i]=='?' or g[i]!=x[i] for i in range(len(attributes))):
                    G_new.append(g)
                else:
                    for i in range(len(attributes)):
                        if g[i]=='?':
                            g_temp = g.copy()
                            g_temp[i] = S[i]
                            if g_temp not in G_new:
                                G_new.append(g_temp)
            G = G_new
            
    print("Specific Boundary (S):", S)
    print("General Boundary (G):", G)

candidate_elimination(data)
