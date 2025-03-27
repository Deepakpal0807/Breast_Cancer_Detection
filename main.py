import pandas as pd
from sklearn.preprocessing import StandardScaler # used for fit and transform the data.
from sklearn.model_selection import train_test_split # used for splitting the data into training and testing sets.
from sklearn.linear_model import LogisticRegression # used for creating a logistic regression model.
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix # used for evaluating the model.
import pickle  # Use the built-in pickle module


def dataclean():
    # Load the data
    data=pd.read_csv("data.csv")
 

    #understanding the data 
    #checking for null value and more..


    # print(data.info())
    # print(data.describe())

    # print(data.isnull().sum())

    #there are 1 column that contain all Nan value..
    #so we drop that column
    # colname =Unnamed: 32

    data.drop(['Unnamed: 32','id'],axis=1,inplace=True)

    # We also drop the id column because it doesn't affect the diagnosis
    # data.drop(['id'],axis=1,inplace=True)

    
    #checking for null value again
    # print(data.isnull().sum())

    # Converting the Diagnosis Column..
    # Convert the M into 1 and B into 0

    data['diagnosis']=data['diagnosis'].map({'M':1,'B':0})
    # print(data.head())

    

    return data

def create_model(data):
    x=data.drop(['diagnosis'],axis=1)
    y=data['diagnosis']

    scalar=StandardScaler()
    x=scalar.fit_transform(x)
    # use means and standard deviation to fit and transform the data..
    # print(x)""

    #split the data for training and testing..
    x_train,x_test,y_train,y_test=train_test_split(
        x,y,test_size=0.2,random_state=42
    )

    #train the model
    model=LogisticRegression()
    model.fit(x_train,y_train)

    #test the model
    y_pred=model.predict(x_test)
    print("Accuracy Score of our model : ",accuracy_score(y_test,y_pred))
    print("Classification Report \n : ",classification_report(y_test,y_pred))
    



    # print(model.score(x_test,y_test))
    # print(model.score(x_train,y_train))
    return model,scalar 
    


def start():
    data=dataclean()

    # print(data.tail())

    model,scalar=create_model(data)

    #export the modell.
    with open("model/model.pkl",'wb') as f:
        pickle.dump(model,f)
    with open("model/scalar.pkl",'wb') as f:
        pickle.dump(scalar,f)




if __name__=='__main__':
    start()
       
    