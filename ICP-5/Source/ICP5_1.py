import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use(style='ggplot')
plt.rcParams['figure.figsize'] = (20, 10)

train = pd.read_csv('C:\\Users\\vidyu\\Desktop\\Temp\\Python_Lesson5\\train.csv')

garage_field = train['GarageArea']
sale_price = train['SalePrice']
plt.scatter(garage_field,sale_price,alpha=.50,color='r')
plt.xlabel('Garage_Field:Column')
plt.ylabel('Sale_Price:Target')
plt.show()

outliers = train['GarageArea']<=1100
train=train[outliers]
outliers = train['GarageArea']!=0
train=train[outliers]

garage_field = train['GarageArea']
sale_price = train['SalePrice']
plt.scatter(garage_field,sale_price,alpha=.50,color='g')
plt.xlabel('Garage_Field:Column')
plt.ylabel('Sale_Price:Target')
plt.show()