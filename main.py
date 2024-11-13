from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import numpy as np
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import accuracy_score,f1_score,recall_score

# P1
bc = load_breast_cancer()
df = pd.DataFrame(bc.data, columns = bc.feature_names)
columns = bc.feature_names

# P2
print("first 10 rows of dataset")
print(df[:10])

# P3
for x in columns:
    print(x," Stats\n")
    print(df[x].describe())
    print("\n")
    
# P5
ss = StandardScaler()
ss.fit_transform(bc.data)
x_train, x_test, y_train, y_test = train_test_split(bc.data,bc.target,test_size = 0.3)


# P7
accuracies = []
for k in range(1,150):
    model = KNeighborsClassifier(n_neighbors = k )
    model.fit(x_train,y_train)
    y_pred = model.predict(x_test)
    accuracies.append((accuracy_score(y_test,y_pred),k))
    
# P8

adf = pd.DataFrame(accuracies, columns = ["accuracy","k_value"])
max_k = adf[adf["accuracy"] == adf.max()["accuracy"]]["k_value"]
print(max_k)

# p9
model = KNeighborsClassifier(n_neighbors = max_k[0] )
model.fit(x_train,y_train)
y_pred = model.predict(x_test)
print(f1_score(y_test,y_pred),recall_score(y_test,y_pred))


# P6
sns.heatmap(df.corr(), cmap = 'hot')
plt.show()

imp = SimpleImputer()
df_imp = pd.DataFrame(imp.fit_transform(df), columns = bc.feature_names)
if(df_imp == df):
    print("Not Missing")
else:
    print("Not Missing")

