import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression

df = pd.read_csv("data.csv")

X = df[["Height", "Weight", "Eye"]]
X = X.replace(["Brown", "Blue"], [1,0])

# print(df.head)

y = df["Species"]

clf = LogisticRegression()
clf.fit(X, y)
joblib.dump(clf, "clf.pkl")

