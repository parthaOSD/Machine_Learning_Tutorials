import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def loss_func(m ,b, points):
    total_loss=0
    for i in range(len(points)):
        x=points.iloc[i]["hours"]
        y=points.iloc[i]["marks"]
        total_loss=total_loss + (y-(m*x + b))**2
    return total_loss
def gradient_decent(new_m, new_b, points,L):
    m_grad=0
    b_grad=0
    n=len(points)
    for i in range(n):
        x=points.iloc[i]["hours"]
        y=points.iloc[i]["marks"]
        m_grad=m_grad-(2/n)*(x*(y-(new_m*x+new_b)))
        b_grad=b_grad-(2/n)*(y-(new_m*x+new_b)) 

    m=new_m-m_grad*L
    b=new_b-b_grad*L
    return m,b

def plot( m,b, points):
    x_value=points["hours"]
    y_value=m*x_value+b
    plt.scatter(points["hours"],points["marks"])
    plt.plot(x_value,y_value,color="red")
    plt.show()

df=pd.read_csv("studentdata(using_numpy).csv")
m=0
b=0
L=0.001
for i in range(100):
    m,b=gradient_decent(m,b,df,L)
plot(m,b,df)