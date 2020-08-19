

```python
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
dataset = pd.read_csv('Data.csv')
```


```python
dataset.head()
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Country</th>
      <th>Age</th>
      <th>Salary</th>
      <th>Purchased</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>France</td>
      <td>44.0</td>
      <td>72000.0</td>
      <td>No</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Spain</td>
      <td>27.0</td>
      <td>48000.0</td>
      <td>Yes</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Germany</td>
      <td>30.0</td>
      <td>54000.0</td>
      <td>No</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Spain</td>
      <td>38.0</td>
      <td>61000.0</td>
      <td>No</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Germany</td>
      <td>40.0</td>
      <td>NaN</td>
      <td>Yes</td>
    </tr>
  </tbody>
</table>
</div>



## X(features) and Y(predict variable) separation from dataset


```python
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 3].values
```

## Dealing with missing data


```python
# replacing missing data with he mean of the colum
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values='NaN', strategy='mean', axis=0)
```


```python
# select the misssing data columns to fit
imputer = imputer.fit(X[:,1:3])
X[:,1:3] = imputer.transform(X[:,1:3])
```


```python
# checkin out the result
X
```




    array([['France', 44.0, 72000.0],
           ['Spain', 27.0, 48000.0],
           ['Germany', 30.0, 54000.0],
           ['Spain', 38.0, 61000.0],
           ['Germany', 40.0, 63777.77777777778],
           ['France', 35.0, 58000.0],
           ['Spain', 38.77777777777778, 52000.0],
           ['France', 48.0, 79000.0],
           ['Germany', 50.0, 83000.0],
           ['France', 37.0, 67000.0]], dtype=object)



## Encoding Categorical Data


```python
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
label_encoder = LabelEncoder()
onehot_enconder = OneHotEncoder(categorical_features=[0])
```

#### 1. categorize country as number values


```python
X[:,0] = label_encoder.fit_transform(X[:,0])
```

#### 2. So the ML algorithm do not thing each country has a bigger value then the other categorize using one hot. A colum for each country will be created with a 0 or 1 value (dummy enconding)


```python
X = onehot_enconder.fit_transform(X).toarray()
```


```python
pd.DataFrame(data=X,  columns=['France', 'Germany','Spain','Age','Salary'])
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>France</th>
      <th>Germany</th>
      <th>Spain</th>
      <th>Age</th>
      <th>Salary</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>44.000000</td>
      <td>72000.000000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>27.000000</td>
      <td>48000.000000</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>30.000000</td>
      <td>54000.000000</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>38.000000</td>
      <td>61000.000000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>40.000000</td>
      <td>63777.777778</td>
    </tr>
    <tr>
      <th>5</th>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>35.000000</td>
      <td>58000.000000</td>
    </tr>
    <tr>
      <th>6</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>38.777778</td>
      <td>52000.000000</td>
    </tr>
    <tr>
      <th>7</th>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>48.000000</td>
      <td>79000.000000</td>
    </tr>
    <tr>
      <th>8</th>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>50.000000</td>
      <td>83000.000000</td>
    </tr>
    <tr>
      <th>9</th>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>37.000000</td>
      <td>67000.000000</td>
    </tr>
  </tbody>
</table>
</div>



### 3. Categorize Y


```python
label_encoder = LabelEncoder()
onehot_enconder = OneHotEncoder(categorical_features=[0])
y = label_encoder.fit_transform(y)
```


```python
pd.DataFrame(data=y,  columns=['No(0)/Yes(1)'])
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>No(0)/Yes(1)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
    </tr>
    <tr>
      <th>5</th>
      <td>1</td>
    </tr>
    <tr>
      <th>6</th>
      <td>0</td>
    </tr>
    <tr>
      <th>7</th>
      <td>1</td>
    </tr>
    <tr>
      <th>8</th>
      <td>0</td>
    </tr>
    <tr>
      <th>9</th>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



## Spliting Dataset into Training and Test sets.


```python
from sklearn.model_selection import train_test_split
```


```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=0)
```


```python
def show_X_y():
    print('X Train:')
    print(pd.DataFrame(data=X_train,  columns=['France', 'Germany','Spain','Age','Salary']))
    print('\nX Test:')
    print(pd.DataFrame(data=X_test,  columns=['France', 'Germany','Spain','Age','Salary']))
    print('\nY Train:')
    print(pd.DataFrame(data=y_train,  columns=['No(0)/Yes(1)']))
    print('\nX Test:')
    print(pd.DataFrame(data=y_test,  columns=['No(0)/Yes(1)']))
    
show_X_y()
```

    X Train:
       France   Germany     Spain       Age    Salary
    0    -1.0  2.645751 -0.774597  0.263068  0.123815
    1     1.0 -0.377964 -0.774597 -0.253501  0.461756
    2    -1.0 -0.377964  1.290994 -1.975398 -1.530933
    3    -1.0 -0.377964  1.290994  0.052614 -1.111420
    4     1.0 -0.377964 -0.774597  1.640585  1.720297
    5    -1.0 -0.377964  1.290994 -0.081312 -0.167514
    6     1.0 -0.377964 -0.774597  0.951826  0.986148
    7     1.0 -0.377964 -0.774597 -0.597881 -0.482149
    
    X Test:
       France   Germany     Spain       Age    Salary
    0    -1.0  2.645751 -0.774597 -1.458829 -0.901663
    1    -1.0  2.645751 -0.774597  1.984964  2.139811
    
    Y Train:
       No(0)/Yes(1)
    0             1
    1             1
    2             1
    3             0
    4             1
    5             0
    6             0
    7             1
    
    X Test:
       No(0)/Yes(1)
    0             0
    1             0
    

## Feature Scaling


```python
from sklearn.preprocessing import StandardScaler
std_scaler_X = StandardScaler()
```


```python
X_train = std_scaler_X.fit_transform(X_train)
X_test = std_scaler_X.transform(X_test) #do not need to fi because it's already fit by X_train
```


```python
show_X_y()
```

    X Train:
       France   Germany     Spain       Age    Salary
    0    -1.0  2.645751 -0.774597  0.263068  0.123815
    1     1.0 -0.377964 -0.774597 -0.253501  0.461756
    2    -1.0 -0.377964  1.290994 -1.975398 -1.530933
    3    -1.0 -0.377964  1.290994  0.052614 -1.111420
    4     1.0 -0.377964 -0.774597  1.640585  1.720297
    5    -1.0 -0.377964  1.290994 -0.081312 -0.167514
    6     1.0 -0.377964 -0.774597  0.951826  0.986148
    7     1.0 -0.377964 -0.774597 -0.597881 -0.482149
    
    X Test:
       France   Germany     Spain       Age    Salary
    0    -1.0  2.645751 -0.774597 -1.458829 -0.901663
    1    -1.0  2.645751 -0.774597  1.984964  2.139811
    
    Y Train:
       No(0)/Yes(1)
    0             1
    1             1
    2             1
    3             0
    4             1
    5             0
    6             0
    7             1
    
    X Test:
       No(0)/Yes(1)
    0             0
    1             0
    


```python

```
