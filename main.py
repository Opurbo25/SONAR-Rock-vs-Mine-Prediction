#import essential libraries
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

#read csv file
df=pd.read_csv("sonar.csv")
df.head()

#checking columns name
col=df.columns

#making input and result
input=df.drop("R",axis=1)
result=df["R"]

#prepocessing R column
model=LabelEncoder()
result=model.fit_transform(result)

#creating train and test data
x_train,x_test,y_train,y_test=train_test_split(input,result,test_size=0.1,random_state=42)

#creating logistic regression model
model=LogisticRegression()
model.fit(x_train,y_train)

#testing model
y=model.predict(x_test)
result=[]
for i in y:
    if i==1:
        result.append('Rock')
    else:
        result.append('Mine')
print(result)

#model accuracy test
s=model.score(x_test,y_test)
print(s)
