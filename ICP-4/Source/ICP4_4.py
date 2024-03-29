
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

# load glass data set
glass = pd.read_csv('C:\\Users\\vidyu\\Desktop\\python code\\glass.csv')
x = glass[['RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe']]
y = glass['Type']


X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.4, random_state=0)
print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)

# Implement SVM method with RBF Kernel using scikit library
svm = SVC(kernel='rbf')
svm.fit(X_train, y_train)
print('Accuracy of SVM classifier on training set: {:.2f}'.format(svm.score(X_train, y_train)))
# test data set acc
print('Accuracy of SVM classifier on test set: {:.2f}'.format(svm.score(X_test, y_test)))
