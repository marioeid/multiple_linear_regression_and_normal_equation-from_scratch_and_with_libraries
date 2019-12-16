import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


           # first requirement training our data set using gradient descent 
def normalEquation(X, y):
    step1 = np.dot(X.T, X)
    step2 = np.linalg.pinv(step1)
    step3 = np.dot(step2, X.T)
    theta = np.dot(step3, y)
    return theta

def Z_O_feature_encoder_negative(X,col):
     rows=len(X)
     for i in range(rows):
        if X.iat[i,col]=='No Negative':
            X.iat[i,col]=0
        else :
            X.iat[i,col]=1
     return X   


def Z_O_feature_encoder_positive(X,col):
     rows=len(X)
     for i in range(rows):
        if X.iat[i,col]=='No Positive':
            X.iat[i,col]=0
        else :
            X.iat[i,col]=1
     return X   
           
    
def feature_encoder(X,cols):
    # function to transform non numric data to numric data
    # be carefull if cols=('') and there's only one value 
    # it won't considered as a one row array 
    
    for c in cols:
       enc = LabelEncoder()
       enc.fit(X[c])
       X[c] = enc.transform(X[c])
    return X;

    
def cost_func(X,y,theta):
    # j(theta)=1/2m*sum for every m(h(theta)-y)^2
    # don't repreduce the code as functions (recursive) cause i tried and it gave me an error
    # m is the number of inputs for one feature
    # X.dot(theta) is the same as theta+X1*theta1+X2*theta2 python multiplicatoins are faster
    m=len(y)
    j= np.sum((X.dot(theta)-y)**2)/(2*m)
    return j

def gradient_descent(X,y,theta,alpha,iterations):
    
    # array to hold the iterations 
    cost_array=[0]*iterations
    
    #the number of inputs for one feature
    m=len(y)
    
    for iteration in range(iterations) :
        # hypothesis h(theta)
        h=X.dot(theta)
        # loss function
        loss=h-y
        # the gradient step 
        # remember 
        # theta =theta-alpha*(1/m)*sum for every m(loss)*x
        # using matrix multiplication is faster 
        dervative=X.T.dot(loss)
        # now our new best thetas we update for every iteration 
        # Ahmed if you did any modification check that the cost is decreasing
        # if the cost is increasing then there's some thing wrong in the 
        # gradient or in the code
        theta-=(alpha*dervative)/m
        cost_array[iteration]=cost_func(X,y,theta)
    return theta,cost_array;        
    
#pre processing data 
data = pd.read_csv('Hotel_Reviews.csv')

#Drop the rows that contain missing values
data.dropna(how='any',inplace=True)

# take all our data and then drop useless features 
X=data.iloc[:,:17] 
X=X.drop('Reviewer_Score',1)

X=X.drop('lat',1)
X=X.drop('lng',1)
X=Z_O_feature_encoder_negative(X,6)
X=Z_O_feature_encoder_positive(X,9)
# make non numeric values numeric so that the pc can under stand it  
cols=('days_since_review','Review_Date','Hotel_Address','Tags','Hotel_Name','Reviewer_Nationality','Negative_Review','Positive_Review');
X=feature_encoder(X,cols);

# taking our predction column and spliting the data  
Y=data['Reviewer_Score']
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.30,shuffle=True)
                 
# feature scalling (normalization)
X_train=(X_train-X_train.min())/(X_train.max()-X_train.min())
y_train=(y_train-y_train.min())/(y_train.max()-y_train.min())

X_test=(X_test-X_test.min())/(X_test.max()-X_test.min())
y_test=(y_test-y_test.min())/(y_test.max()-y_test.min())
                 # using the model as functions not from scratch
#Top 50% Correlation training features with the Value

cls=linear_model.LinearRegression()
cls.fit(X_train,y_train) 
prediction_cls_train= cls.predict(X_train)
prediction_cls_test=cls.predict(X_test)
mse_train_cls_model=metrics.mean_squared_error(y_train,prediction_cls_train)
mse_test_cls_model=metrics.mean_squared_error(y_test,prediction_cls_test)
       # adding additional one feature to the begging of our training and testing set 

x0=np.ones(len(y_train))
x0test=np.ones(len(y_test))
X_train.insert(loc=0,column="X0",value=x0)
X_test.insert(loc=0,column="X0",value=x0test)

# making our theta a row with the same size of features for matrix multiplication
# intial theta is zero so we make our first predction of features equal to zero 
theta=np.zeros(15)

# gradient descent to predict cost and thetas (it gave Nan at first due non normalized data)
# that's why i used feature scalling 

# the best parametrs for thetas and the cost for every iteration 
best_thetas,cost_array=gradient_descent(X_train,y_train,theta,0.1,100)

# the produced predction for the training and test set 
Y_pred_for_training_set=X_train.dot(best_thetas)
Y_pred_for_test_set=X_test.dot(best_thetas)
mse_train_my_model=metrics.mean_squared_error(y_train,Y_pred_for_training_set)
mse_test_my_model=metrics.mean_squared_error(y_test,Y_pred_for_test_set)

                       # second requirement using normal equation
theta=normalEquation(X_train,y_train)
prediction_normal_equation=X_test.dot(theta)
mse_normal_equation=metrics.mean_squared_error(y_test,prediction_normal_equation)
                       