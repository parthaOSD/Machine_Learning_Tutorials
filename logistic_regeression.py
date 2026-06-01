import numpy as np

def sigmoid(z):
    return 1/(1+np.exp(-z))

def gradient(theta, X,y):
    m=y.size
    h=sigmoid(X@theta)
    return (X.T@(h-y))/m

def gradient_descent(X,y,num_iteration=100, tol=1e-7,alpha=0.1):
    X_b=np.c_[np.ones((X.shape[0],1)), X]
    theta=np.zeros(X_b.shape[1])
    for i in range(num_iteration):
        grad=gradient(theta, X_b,y)
        theta=theta-alpha*grad
        if np.linalg.norm(grad)<tol:
            break
    return theta

def predict_probability(X,theta):
    X_b=np.c_[np.ones((X.shape[0],1)), X]
    return sigmoid(X_b@theta)

def predict(X,theta, threshold=0.5):
    return (predict_probability(X,theta)>=threshold).astype(int)

from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

X,y=load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test=train_test_split(X,y, test_size=0.2)
scaler=StandardScaler()
X_train_scaled=scaler.fit_transform(X_train)
X_test_scaled=scaler.transform(X_test)
theta_hat=gradient_descent(X_train_scaled, y_train)
y_pred_train=predict(X_train_scaled,theta_hat)
y_pred_test=predict(X_test_scaled,theta_hat)
train_acc=accuracy_score(y_train, y_pred_train)
test_acc=accuracy_score(y_test,y_pred_test)
print(train_acc)
print(test_acc)
      

     
    
