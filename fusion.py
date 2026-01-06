import pandas as pd
exercise = pd.read_csv("exercise.csv")
calories = pd.read_csv("calories.csv")

data_fusion = pd.merge(exercise, calories, on="User_ID")
print(data_fusion.head())

data_fusion.to_csv("data_fusion.csv", index=False)
