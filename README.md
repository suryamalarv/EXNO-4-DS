# EXNO:4-DS
# AIM:
To read the given data and perform Feature Scaling and Feature Selection process and save the
data to a file.

# ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Scaling for the feature in the data set.
STEP 4:Apply Feature Selection for the feature in the data set.
STEP 5:Save the data to the file.

# FEATURE SCALING:
1. Standard Scaler: It is also called Z-score normalization. It calculates the z-score of each value and replaces the value with the calculated Z-score. The features are then rescaled with x̄ =0 and σ=1
2. MinMaxScaler: It is also referred to as Normalization. The features are scaled between 0 and 1. Here, the mean value remains same as in Standardization, that is,0.
3. Maximum absolute scaling: Maximum absolute scaling scales the data to its maximum value; that is,it divides every observation by the maximum value of the variable.The result of the preceding transformation is a distribution in which the values vary approximately within the range of -1 to 1.
4. RobustScaler: RobustScaler transforms the feature vector by subtracting the median and then dividing by the interquartile range (75% value — 25% value).

# FEATURE SELECTION:
Feature selection is to find the best set of features that allows one to build useful models. Selecting the best features helps the model to perform well.
The feature selection techniques used are:
1.Filter Method
2.Wrapper Method
3.Embedded Method

# CODING AND OUTPUT:
```
DEVELOPED BY: SURYAMALARV
REGISTER NO:  212223230224
```
```
import pandas as pd
from scipy import stats
import numpy as np
df=pd.read_csv("/content/bmi.csv")
df.head()
```
![image](https://github.com/user-attachments/assets/8dcc1d01-32af-406f-a594-df4fc9770245)

```
df.dropna()
```
![image](https://github.com/user-attachments/assets/ebc918b4-e0e6-4750-9374-5beb35ce08ec)

```
max_vals=np.max(np.abs(df[['Height','Weight']]))
max_vals
```
![image](https://github.com/user-attachments/assets/d181c8f9-13f6-4da6-9249-d5c1d337eaa7)

```
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
df[['Height','Weight']]=sc.fit_transform(df[['Height','Weight']])
df.head(10)
```
![image](https://github.com/user-attachments/assets/bda64e97-07fc-451c-baf8-240f6b95d669)

```
from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()
df[['Height','Weight']]=scaler.fit_transform(df[['Height','Weight']])
df.head(10)
```
![image](https://github.com/user-attachments/assets/4ca7161b-0c2e-48bd-8813-3aed3d9275b9)

```
from sklearn.preprocessing import Normalizer
scaler=Normalizer()
df[['Height','Weight']]=scaler.fit_transform(df[['Height','Weight']])
df
```
![image](https://github.com/user-attachments/assets/577179c6-9b75-43d6-9ed4-d93f9e313005)

```
from sklearn.preprocessing import MaxAbsScaler
scaler=MaxAbsScaler()
df[['Height','Weight']]=scaler.fit_transform(df[['Height','Weight']])
df
```
![image](https://github.com/user-attachments/assets/9549def8-b384-4551-bb81-aa57c4d8d4df)
```
from sklearn.preprocessing import RobustScaler
scaler=RobustScaler()
df[['Height','Weight']]=scaler.fit_transform(df[['Height','Weight']])
df.head()
```
![image](https://github.com/user-attachments/assets/4fc29bcf-d691-4ec0-915f-43268802e66c)

```
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score,confusion_matrix
data = pd.read_csv('/content/income(1) (1).csv',na_values=[" ?"])
data
```
![image](https://github.com/user-attachments/assets/3ed2e2f2-2811-49bf-93c9-48d50898712f)
```
data.isnull().sum()
```
![image](https://github.com/user-attachments/assets/62f2278a-4bdc-4f96-b010-fecf9e520e65)

```
missing = data[data.isnull().any(axis=1)]
missing
```
![image](https://github.com/user-attachments/assets/3dbb527e-1d0d-4050-b639-28aa7d86b9df)

```
data2 = data.dropna(axis=0)
data2
```
![image](https://github.com/user-attachments/assets/98918c10-5d01-4a52-a6f2-52b7f5c3f9a9)
```
sal=data['SalStat']
data2['SalStat']=data2['SalStat'].map({' less than or equal to 50,000':0,' greater than 50,000':1})
print(data2['SalStat'])
```
![image](https://github.com/user-attachments/assets/c3970039-d6ed-47a7-a04c-331111a5f017)
```

sal2=data2['SalStat']
dfs=pd.concat([sal,sal2],axis=1)
dfs
```
![image](https://github.com/user-attachments/assets/f146cf38-bf1b-4f92-a45e-ca26abff2355)

```
data2
```
![image](https://github.com/user-attachments/assets/b1a2ae36-45c5-4525-9cfd-8b1e9b4dde84)

```
new_data=pd.get_dummies(data2,drop_first=True)
new_data
```
![image](https://github.com/user-attachments/assets/d1151bc9-dc91-474c-a22b-21792654422c)

```
y=new_data['SalStat'].values
print(y)
```
![image](https://github.com/user-attachments/assets/6d4583a2-4872-4b1c-80f4-1424082d2f47)

```
train_x,test_x,train_y,test_y=train_test_split(x,y,test_size=0.3,random_state=0)
KNN_classifier=KNeighborsClassifier(n_neighbors=5)
KNN_classifier.fit(train_x,train_y)
```
![image](https://github.com/user-attachments/assets/1002750a-8fad-43a9-a801-8ea418447039)

```
prediction=KNN_classifier.predict(test_x)
confusionMmatrix=confusion_matrix(test_y,prediction)
print(confusionMatrix)
```
![image](https://github.com/user-attachments/assets/376b284f-f7c0-4dd0-ba1b-7398ecf8983f)

```
import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency
import seaborn as sns
tips=sns.load_dataset('tips')
tips.head()
```
![image](https://github.com/user-attachments/assets/73ea2637-e30e-4c17-835e-f75cb352f401)

```
chi2,p,_,_=chi2_contingency(contingency_table)
print(f"Chi-Square Statistic: {chi2}")
print(f"p-value: {p}")
```
![image](https://github.com/user-attachments/assets/585e772b-121f-4c58-9ddf-00db7a3f5330)

```
X=df[['Feature1','Feature3']]
y=df['Target']
selector=SelectKBest(score_func=mutual_info_classif,k=1)
X_new=selector.fit_transform(X,y)
selected_features_indices=selector.get_support(indices=True)
selected_features=X.columns[selected_features_indices]
print("Selected Features:")
print(selected_features)
```
![image](https://github.com/user-attachments/assets/82ea2bd6-a2a7-470c-b33d-e726bbcf1810)
   
# RESULT:

      Thus the code for Feature Scaling and Feature Selection process has been executed.
