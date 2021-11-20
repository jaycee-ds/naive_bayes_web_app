import pandas as pd
import matplotlib.pyplot as plt
from sklearn.naive_bayes import CategoricalNB
from sklearn.metrics import classification_report, confusion_matrix
import pickle

df = pd.read_csv('Generated_data.csv')
print(df.head())

for col in df.columns:
    plt.hist(df[col])
    plt.title(f'Distribution of {col}')
    plt.savefig(f'Distribution of {col}.png')
    plt.close()

df.replace({'ad':1, 'not an ad':2,
            'phishing':1, 'not phishing':2,
            'unknown':1, 'contact':2},
           inplace=True)
train = df[:int(0.7*len(df))]
test = df[int(0.7*len(df)):]

X_train = train.drop('filter', axis=1)
y_train = train['filter']
X_test = test.drop('filter', axis=1)
y_test = test['filter']

catNB = CategoricalNB()
model = catNB.fit(X_train, y_train)

predictions = model.predict(X_test)

clf_report = classification_report(y_test, predictions)
conf_matrix = confusion_matrix(y_test, predictions)
print(clf_report)
print(conf_matrix)

with open('Naive Bayes model.pkl', 'wb') as f:
    pickle.dump(model, f)
    print('Pickling completed.')