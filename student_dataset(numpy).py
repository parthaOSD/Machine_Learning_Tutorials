import numpy as np
import pandas as pd

hours=np.array([1,2,4,6])
marks= np.array([20,40,50,70])

df=pd.DataFrame({
    "hours":[1,2,4,6],
    "marks":[20,40,50,70]
})
df.to_csv("studentdata(using_numpy).csv", index=False)
