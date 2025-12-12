from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report

categories = ['alt.atheism', 'sci.space', 'talk.religion.misc', 'rec.autos']
train = fetch_20newsgroups(subset='train', categories=categories, shuffle=True)
test = fetch_20newsgroups(subset='test', categories=categories, shuffle=True)

text_clf = Pipeline([
    ('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('clf', MultinomialNB())
])

text_clf.fit(train.data, train.target)
predicted = text_clf.predict(test.data)

print("Accuracy:", accuracy_score(test.target, predicted))
print("Precision:", precision_score(test.target, predicted, average='macro'))
print("Recall:", recall_score(test.target, predicted, average='macro'))
print("Classification Report:\n", classification_report(test.target, predicted, target_names=test.target_names))
