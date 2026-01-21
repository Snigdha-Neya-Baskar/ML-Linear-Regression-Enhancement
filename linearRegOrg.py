import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

# Sample dataset
data = {
    'Hours': [1, 2, 3, 4, 5],
    'Marks': [35, 40, 50, 60, 75]
}

df = pd.DataFrame(data)

X = df[['Hours']]
y = df['Marks']

model = LinearRegression()
model.fit(X, y)

predicted = model.predict([[6]])
print("Predicted marks for 6 hours:", predicted[0])
