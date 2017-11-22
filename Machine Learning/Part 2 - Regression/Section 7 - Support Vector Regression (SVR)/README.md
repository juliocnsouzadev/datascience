
___
# Support Vector Regression
___

## Imports


```python
# data analysis
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# ploting
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
plt.style.use('fivethirtyeight')
plt.rcParams['figure.figsize'] = (15,5)

# data preprocesing
from sklearn.preprocessing import Imputer #replace missing data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder #encoding categorical data
from sklearn.preprocessing import StandardScaler #feature scaling

#spliting data
from sklearn.model_selection import train_test_split

# models
from sklearn.svm import SVR

# metrics
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from math import sqrt
```

## Exploring the Dataset


```python
dataset = pd.read_csv("Position_Salaries.csv")
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
      <th>Position</th>
      <th>Level</th>
      <th>Salary</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Business Analyst</td>
      <td>1</td>
      <td>45000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Junior Consultant</td>
      <td>2</td>
      <td>50000</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Senior Consultant</td>
      <td>3</td>
      <td>60000</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Manager</td>
      <td>4</td>
      <td>80000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Country Manager</td>
      <td>5</td>
      <td>110000</td>
    </tr>
  </tbody>
</table>
</div>




```python
dataset.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 10 entries, 0 to 9
    Data columns (total 3 columns):
    Position    10 non-null object
    Level       10 non-null int64
    Salary      10 non-null int64
    dtypes: int64(2), object(1)
    memory usage: 320.0+ bytes
    


```python
dataset.describe()
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
      <th>Level</th>
      <th>Salary</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>10.00000</td>
      <td>10.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>5.50000</td>
      <td>249500.000000</td>
    </tr>
    <tr>
      <th>std</th>
      <td>3.02765</td>
      <td>299373.883668</td>
    </tr>
    <tr>
      <th>min</th>
      <td>1.00000</td>
      <td>45000.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>3.25000</td>
      <td>65000.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>5.50000</td>
      <td>130000.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>7.75000</td>
      <td>275000.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>10.00000</td>
      <td>1000000.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
sns.heatmap(dataset.corr(), cmap="coolwarm")
```




    <matplotlib.axes._subplots.AxesSubplot at 0x1d72a2903c8>




![png](output_8_1.png)


## Dataprocesing


```python
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values
```

### Featuring Scaling


```python
scaler_X = StandardScaler()
scaler_y = StandardScaler()
X = scaler_X.fit_transform(X)
y = scaler_y.fit_transform(y.reshape(-1, 1))
```

    C:\ProgramData\Anaconda3\lib\site-packages\sklearn\utils\validation.py:475: DataConversionWarning: Data with input dtype int64 was converted to float64 by StandardScaler.
      warnings.warn(msg, DataConversionWarning)
    

## Creating and Training the model


```python
regressor = SVR(kernel='rbf')
regressor.fit(X,y)
```

    C:\ProgramData\Anaconda3\lib\site-packages\sklearn\utils\validation.py:578: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples, ), for example using ravel().
      y = column_or_1d(y, warn=True)
    




    SVR(C=1.0, cache_size=200, coef0=0.0, degree=3, epsilon=0.1, gamma='auto',
      kernel='rbf', max_iter=-1, shrinking=True, tol=0.001, verbose=False)



## Perfomance


```python
def performance_metric(y_true, y_predict):
    score = r2_score(y_true, y_predict)
    return score

def plot_performance(X, y, regressor, model_name):
    plt.scatter(X, y, color = 'red')
    y_predict = regressor.predict(X)
    plt.plot(X, y_predict, color = 'blue')
    plt.xlabel('Position level')
    plt.ylabel('Salary')
    score = performance_metric(y,  y_predict)
    print ("\nThe " + model_name +" model R^2, is {:.3f}.".format(score))
    plt.title("The " + model_name +" model R^2, is {:.3f}.".format(score))
    plt.show()
```


```python
plot_performance(X,y,regressor,"SVR")
```

    
    The SVR model R^2, is 0.752.
    


![png](output_17_1.png)



```python
arr= np.array([[6.5]])
scaled_value = scaler_X.transform(arr)
not_scaled_predict = regressor.predict(scaled_value)
scaler_y.inverse_transform(not_scaled_predict)[0]
```




    170370.02040650236


