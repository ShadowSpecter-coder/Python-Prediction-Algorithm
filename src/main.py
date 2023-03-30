import pandas as pd
from sklearn.tree import DecisionTreeClassifier

data = pd.read_csv('../assets/music.csv')
input_set = data.drop(columns=['genre'])
output_set = data['genre']

model = DecisionTreeClassifier()
model.fit(input_set, output_set)
prediction = model.predict([[28, 0], [35, 1]])
print(prediction)
    