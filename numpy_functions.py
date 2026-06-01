import numpy as np
arr=np.array([[1,2,3,4],
              [3,5,6,7],
              [4,6,7,8]])
def sum_of_array(arr, indices):
    total=0
    for i,j in indices:
        total=total+arr[i,j]
    return total
print("Sum of array:",sum_of_array(arr, [(1,2),(1,3),(2,3)]))
    



    
  

