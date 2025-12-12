import pandas as pd
import numpy as np

df= pd.read_csv(r"C:\Users\kvami\OneDrive\Desktop\coding\machine_learning\test_dataset.csv")

print(df.head(20))
print(df.shape)
print(df.info())
print(df.describe())

x=df[['Age','Salary','Experience']]
y=df['Purchased']

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

from sklearn.linear_model import LogisticRegression
model=LogisticRegression()
model.fit(x_train,y_train)

y_pred=model.predict(x_test)        

from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
print("Accuracy:",accuracy_score(y_test,y_pred))        
print("Confusion Matrix:\n",confusion_matrix(y_test,y_pred))
print("Classification Report:\n",classification_report(y_test,y_pred))

new_data = [[30, 50000, 5]]   # Age=30, Salary=50k, Exp=5 years
print("Prediction:", model.predict(new_data))
