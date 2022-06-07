from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

#read csv
data = pd.read_csv("diabetes.csv")

#train
x = data.drop('Outcome',axis = 1).values
y = data['Outcome'].values
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.4,random_state=0)

#model
knn = KNeighborsClassifier(n_neighbors=8)

#test model
knn.fit(x_train,y_train)

#predict
predict = knn.predict(x_test)

#create def fuction for set i number
def pred():
    i = int(input('the number of data :'))
    if predict[i] == 0:
        print('Not Diabetic pateint')
    else :
        print('Diabetic pateint')
    print('predict =',predict[i])
    print('y_test = ',y_test[i])
    print('x_test = ',x_test[i])

pred()
print('accuracy = ',accuracy_score(y_test,predict)*100)
