import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
df= pd.read_csv(r"C:\Users\kvami\OneDrive\Desktop\coding\machine_learning\test_dataset.csv")

print(df.head(20))
print(df.shape)
print(df.info())
print(df.describe())

sns.countplot(x='Purchased', data=df)

plt.show()

counts = df['Purchased'].value_counts()
labels = ['Not Purchased', 'Purchased']   # 0 = Not Purchased, 1 = Purchased
sizes = counts.values

plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
plt.title("Purchased Pie Chart")
plt.show()


plt.hist(df['Age'])
plt.show()

sns.distplot(df['Age'])
plt.show()


#MULTIVARIATE ANALYSIS

sns.scatterplot(x='Age', y='Salary', hue='Purchased', data=df)
plt.show()

from pandas_profiling import ProfileReprort
prof= ProfileReprort(df)
prof.to_file(output_file="output.html")