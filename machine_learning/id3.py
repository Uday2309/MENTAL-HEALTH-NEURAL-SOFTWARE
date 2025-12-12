import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_text


data = {
    "Sky": ["Sunny", "Sunny", "Rainy", "Sunny"],
    "AirTemp": ["Warm", "Warm", "Cold", "Warm"],
    "Humidity": ["Normal", "High", "High", "High"],
    "Wind": ["Strong", "Strong", "Strong", "Strong"],
    "Water": ["Warm", "Warm", "Warm", "Cool"],
    "Forecast": ["Same", "Same", "Change", "Change"],
    "EnjoySport": ["Yes", "Yes", "No", "Yes"]
}

df = pd.DataFrame(data)


X = pd.get_dummies(df.drop("EnjoySport", axis=1))
y = df["EnjoySport"].map({"Yes":1, "No":0})


model = DecisionTreeClassifier(criterion='entropy')
model.fit(X, y)


print(export_text(model, feature_names=list(X.columns)))

new_sample = pd.DataFrame([{
    "Sky": "Overcast",
    "AirTemp": "Warm",
    "Humidity": "High",
    "Wind": "Strong",
    "Water": "Cool",
    "Forecast": "Same"
}])


new_sample_encoded = pd.get_dummies(new_sample)
new_sample_encoded = new_sample_encoded.reindex(columns=X.columns, fill_value=0)


prediction = model.predict(new_sample_encoded)
print("\nPrediction for new sample:", "Yes" if prediction[0]==1 else "No")
