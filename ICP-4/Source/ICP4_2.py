
from sklearn.naive_bayes import GaussianNB
import pandas as pd
from sklearn.model_selection import train_test_split


# load glass data set
glass = pd.read_csv('C:\\Users\\vidyu\\Desktop\\python code\\glass.csv')
x = glass[['RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe']]
y = glass['Type']


X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.4, random_state=0)
print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)

clf = GaussianNB()
clf.fit(X_train, y_train)
print('Accuracy of Naive Bayes GaussianNB on training set: {:.2f}'.format(clf.score(X_train, y_train)))
print('Accuracy of Naive Bayes GaussianNB on test set: {:.2f}'.format(clf.score(X_test, y_test)))