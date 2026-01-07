import pandas as pd

data = pd.read_csv("data_fusion.csv")

print(data.head())
print(data.info())
print(data.isnull().sum())

data = data.dropna()

print(data.shape)
data = data.drop_duplicates()
print(data.shape)

data["Gender"] = data["Gender"].map({"male": 0, "female": 1})

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 1000)

print(data.head(10))

for index, row in data.head(10).iterrows():
    print(row.to_dict())

data.to_csv("cleansepare.csv", index=False, sep=';')

