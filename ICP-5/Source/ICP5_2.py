import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('winequality-red.csv')

dataset.quality.describe()

#Working with Numeric Features
numeric_features = dataset.select_dtypes(include=[np.number])

corr = numeric_features.corr()

print (corr['quality'].sort_values(ascending=False)[:4], '\n')
#print (corr['quality'].sort_values(ascending=False)[-5:])

##Null values
nulls = pd.DataFrame(dataset.isnull().sum().sort_values(ascending=False)[:12])
nulls.columns = ['Null Count']
nulls.index.name = 'Feature'
print(nulls)

##handling missing value
data = dataset.select_dtypes(include=[np.number]).interpolate().dropna()
print(sum(data.isnull().sum() != 0))

##Build a linear model
y = np.log(dataset.quality)
X = data.drop(['quality', 'alcohol','sulphates'], axis=1)

#X = data.drop(['fixed acidity','volatile acidity','citric acid','residual sugar','chlorides','free sulfur dioxide','total sulfur dioxide','density','pH','sulphates','alcohol'], axis=1)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=.33)
from sklearn import linear_model
lr = linear_model.LinearRegression()
model = lr.fit(X_train, y_train)

##Evaluate the performance and visualize results
print ("R^2 is: \n", model.score(X_test, y_test))
predictions = model.predict(X_test)
from sklearn.metrics import mean_squared_error
print ('RMSE is: \n', mean_squared_error(y_test, predictions))

##visualize

actual_values = y_test
plt.scatter(predictions, actual_values, alpha=.75, color='g')
plt.xlabel('Predicted Quality')
plt.ylabel('Actual Quality')
plt.title('Regression Model')
plt.show()