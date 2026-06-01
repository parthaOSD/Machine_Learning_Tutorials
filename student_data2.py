import pandas as pd

data={
    "hours":[1,2,4,5],
    "marks":[20,40,60,70]
}

df=pd.DataFrame(data)
df.to_csv("data.csv",index=False)