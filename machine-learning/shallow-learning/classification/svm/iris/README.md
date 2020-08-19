
___

___
# Support Vector Machines Project Iris

Welcome to your Support Vector Machine Project! Just follow along with the notebook and instructions below. We will be analyzing the famous iris data set!

## The Data
For this series of lectures, we will be using the famous [Iris flower data set](http://en.wikipedia.org/wiki/Iris_flower_data_set). 

The Iris flower data set or Fisher's Iris data set is a multivariate data set introduced by Sir Ronald Fisher in the 1936 as an example of discriminant analysis. 

The data set consists of 50 samples from each of three species of Iris (Iris setosa, Iris virginica and Iris versicolor), so 150 total samples. Four features were measured from each sample: the length and the width of the sepals and petals, in centimeters.

Here's a picture of the three different Iris types:


```python
# The Iris Setosa
from IPython.display import Image
url = 'http://upload.wikimedia.org/wikipedia/commons/5/56/Kosaciec_szczecinkowaty_Iris_setosa.jpg'
Image(url,width=300, height=300)
```




![jpeg](output_1_0.jpeg)




```python
# The Iris Versicolor
from IPython.display import Image
url = 'http://upload.wikimedia.org/wikipedia/commons/4/41/Iris_versicolor_3.jpg'
Image(url,width=300, height=300)
```




![jpeg](output_2_0.jpeg)




```python
# The Iris Virginica
from IPython.display import Image
url = 'http://upload.wikimedia.org/wikipedia/commons/9/9f/Iris_virginica.jpg'
Image(url,width=300, height=300)
```

    IOPub data rate exceeded.
    The notebook server will temporarily stop sending output
    to the client in order to avoid crashing it.
    To change this limit, set the config variable
    `--NotebookApp.iopub_data_rate_limit`.
    

The iris dataset contains measurements for 150 iris flowers from three different species.

The three classes in the Iris dataset:

    Iris-setosa (n=50)
    Iris-versicolor (n=50)
    Iris-virginica (n=50)

The four features of the Iris dataset:

    sepal length in cm
    sepal width in cm
    petal length in cm
    petal width in cm

## Get the data

**Use seaborn to get the iris data by using: iris = sns.load_dataset('iris') **


```python
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
%matplotlib inline
plt.style.use('fivethirtyeight')
plt.rcParams['figure.figsize'] = (15,5)

import seaborn as sns
iris = sns.load_dataset('iris')
```

Let's visualize the data and get you started!

## Exploratory Data Analysis


```python
iris.head()
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
      <th>sepal_length</th>
      <th>sepal_width</th>
      <th>petal_length</th>
      <th>petal_width</th>
      <th>species</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>5.1</td>
      <td>3.5</td>
      <td>1.4</td>
      <td>0.2</td>
      <td>setosa</td>
    </tr>
    <tr>
      <th>1</th>
      <td>4.9</td>
      <td>3.0</td>
      <td>1.4</td>
      <td>0.2</td>
      <td>setosa</td>
    </tr>
    <tr>
      <th>2</th>
      <td>4.7</td>
      <td>3.2</td>
      <td>1.3</td>
      <td>0.2</td>
      <td>setosa</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4.6</td>
      <td>3.1</td>
      <td>1.5</td>
      <td>0.2</td>
      <td>setosa</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5.0</td>
      <td>3.6</td>
      <td>1.4</td>
      <td>0.2</td>
      <td>setosa</td>
    </tr>
  </tbody>
</table>
</div>



** Create a pairplot of the data set. Which flower species seems to be the most separable?**


```python
sns.pairplot(data=iris, hue='species')
```




    <seaborn.axisgrid.PairGrid at 0x1a22ba25898>




![png](output_9_1.png)


# Train Test Split

** Split your data into a training set and a testing set.**


```python
from sklearn.cross_validation import train_test_split
```


```python
y = iris['species']
```


```python
X = iris.drop('species', axis=1) 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.3, random_state=42)
```

# Train a Model

Now its time to train a Support Vector Machine Classifier. 

**Call the SVC() model from sklearn and fit the model to the training data.**


```python
from sklearn.svm import SVC
```


```python
classifier = SVC()
```


```python
classifier.fit(X_train, y_train)
```




    SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
      decision_function_shape='ovr', degree=3, gamma='auto', kernel='rbf',
      max_iter=-1, probability=False, random_state=None, shrinking=True,
      tol=0.001, verbose=False)



## Model Evaluation

**Now get predictions from the model and create a confusion matrix and a classification report.**


```python
predictions = classifier.predict(X_test)
predictions
```




    array(['versicolor', 'setosa', 'virginica', 'versicolor', 'versicolor',
           'setosa', 'versicolor', 'virginica', 'versicolor', 'versicolor',
           'virginica', 'setosa', 'setosa', 'setosa', 'setosa', 'versicolor',
           'virginica', 'versicolor', 'versicolor', 'virginica', 'setosa',
           'virginica', 'setosa', 'virginica', 'virginica', 'virginica',
           'virginica', 'virginica', 'setosa', 'setosa', 'setosa', 'setosa',
           'versicolor', 'setosa', 'setosa', 'virginica', 'versicolor',
           'setosa', 'setosa', 'setosa', 'virginica', 'versicolor',
           'versicolor', 'setosa', 'setosa'], dtype=object)




```python
from sklearn.metrics import confusion_matrix, classification_report
```


```python
cm = confusion_matrix(y_true=y_test, y_pred= predictions)
cm
```




    array([[19,  0,  0],
           [ 0, 13,  0],
           [ 0,  0, 13]], dtype=int64)




```python
print(classification_report(y_test, predictions))
```

                 precision    recall  f1-score   support
    
         setosa       1.00      1.00      1.00        19
     versicolor       1.00      1.00      1.00        13
      virginica       1.00      1.00      1.00        13
    
    avg / total       1.00      1.00      1.00        45
    
    

Wow! You should have noticed that your model was pretty good! Let's see if we can tune the parameters to try to get even better (unlikely, and you probably would be satisfied with these results in real like because the data set is quite small, but I just want you to practice using GridSearch.

## Gridsearch Practice

** Import GridsearchCV from SciKit Learn.**


```python
from sklearn.grid_search import GridSearchCV
```

**Create a dictionary called param_grid and fill out some parameters for C and gamma.**


```python
param_grid = {'C':[0.1,1,10,100], 'gamma':[1,0.1,0.01,0.001]}
```

** Create a GridSearchCV object and fit it to the training data.**


```python
grid_search = GridSearchCV(SVC(), param_grid, verbose=2)
grid_search.fit(X_train, y_train)
```

    Fitting 3 folds for each of 16 candidates, totalling 48 fits
    [CV] C=0.1, gamma=1 ..................................................
    [CV] ......................................... C=0.1, gamma=1 -   0.0s
    [CV] C=0.1, gamma=1 ..................................................
    [CV] ......................................... C=0.1, gamma=1 -   0.0s
    [CV] C=0.1, gamma=1 ..................................................
    [CV] ......................................... C=0.1, gamma=1 -   0.0s
    [CV] C=0.1, gamma=0.1 ................................................
    [CV] ....................................... C=0.1, gamma=0.1 -   0.0s
    [CV] C=0.1, gamma=0.1 ................................................
    [CV] ....................................... C=0.1, gamma=0.1 -   0.0s
    [CV] C=0.1, gamma=0.1 ................................................
    [CV] ....................................... C=0.1, gamma=0.1 -   0.0s
    [CV] C=0.1, gamma=0.01 ...............................................
    [CV] ...................................... C=0.1, gamma=0.01 -   0.0s
    [CV] C=0.1, gamma=0.01 ...............................................
    [CV] ...................................... C=0.1, gamma=0.01 -   0.0s
    [CV] C=0.1, gamma=0.01 ...............................................
    [CV] ...................................... C=0.1, gamma=0.01 -   0.0s
    [CV] C=0.1, gamma=0.001 ..............................................
    [CV] ..................................... C=0.1, gamma=0.001 -   0.0s
    [CV] C=0.1, gamma=0.001 ..............................................
    [CV] ..................................... C=0.1, gamma=0.001 -   0.0s
    [CV] C=0.1, gamma=0.001 ..............................................
    [CV] ..................................... C=0.1, gamma=0.001 -   0.0s
    [CV] C=1, gamma=1 ....................................................
    [CV] ........................................... C=1, gamma=1 -   0.0s
    [CV] C=1, gamma=1 ....................................................
    [CV] ........................................... C=1, gamma=1 -   0.0s
    

    [Parallel(n_jobs=1)]: Done   1 out of   1 | elapsed:    0.0s remaining:    0.0s
    

    [CV] C=1, gamma=1 ....................................................
    [CV] ........................................... C=1, gamma=1 -   0.0s
    [CV] C=1, gamma=0.1 ..................................................
    [CV] ......................................... C=1, gamma=0.1 -   0.0s
    [CV] C=1, gamma=0.1 ..................................................
    [CV] ......................................... C=1, gamma=0.1 -   0.0s
    [CV] C=1, gamma=0.1 ..................................................
    [CV] ......................................... C=1, gamma=0.1 -   0.0s
    [CV] C=1, gamma=0.01 .................................................
    [CV] ........................................ C=1, gamma=0.01 -   0.0s
    [CV] C=1, gamma=0.01 .................................................
    [CV] ........................................ C=1, gamma=0.01 -   0.0s
    [CV] C=1, gamma=0.01 .................................................
    [CV] ........................................ C=1, gamma=0.01 -   0.0s
    [CV] C=1, gamma=0.001 ................................................
    [CV] ....................................... C=1, gamma=0.001 -   0.0s
    [CV] C=1, gamma=0.001 ................................................
    [CV] ....................................... C=1, gamma=0.001 -   0.1s
    [CV] C=1, gamma=0.001 ................................................
    [CV] ....................................... C=1, gamma=0.001 -   0.0s
    [CV] C=10, gamma=1 ...................................................
    [CV] .......................................... C=10, gamma=1 -   0.0s
    [CV] C=10, gamma=1 ...................................................
    [CV] .......................................... C=10, gamma=1 -   0.0s
    [CV] C=10, gamma=1 ...................................................
    [CV] .......................................... C=10, gamma=1 -   0.0s
    [CV] C=10, gamma=0.1 .................................................
    [CV] ........................................ C=10, gamma=0.1 -   0.0s
    [CV] C=10, gamma=0.1 .................................................
    [CV] ........................................ C=10, gamma=0.1 -   0.0s
    [CV] C=10, gamma=0.1 .................................................
    [CV] ........................................ C=10, gamma=0.1 -   0.0s
    [CV] C=10, gamma=0.01 ................................................
    [CV] ....................................... C=10, gamma=0.01 -   0.0s
    [CV] C=10, gamma=0.01 ................................................
    [CV] ....................................... C=10, gamma=0.01 -   0.0s
    [CV] C=10, gamma=0.01 ................................................
    [CV] ....................................... C=10, gamma=0.01 -   0.0s
    [CV] C=10, gamma=0.001 ...............................................
    [CV] ...................................... C=10, gamma=0.001 -   0.0s
    [CV] C=10, gamma=0.001 ...............................................
    [CV] ...................................... C=10, gamma=0.001 -   0.0s
    [CV] C=10, gamma=0.001 ...............................................
    [CV] ...................................... C=10, gamma=0.001 -   0.0s
    [CV] C=100, gamma=1 ..................................................
    [CV] ......................................... C=100, gamma=1 -   0.0s
    [CV] C=100, gamma=1 ..................................................
    [CV] ......................................... C=100, gamma=1 -   0.0s
    [CV] C=100, gamma=1 ..................................................
    [CV] ......................................... C=100, gamma=1 -   0.0s
    [CV] C=100, gamma=0.1 ................................................
    [CV] ....................................... C=100, gamma=0.1 -   0.0s
    [CV] C=100, gamma=0.1 ................................................
    [CV] ....................................... C=100, gamma=0.1 -   0.0s
    [CV] C=100, gamma=0.1 ................................................
    [CV] ....................................... C=100, gamma=0.1 -   0.0s
    [CV] C=100, gamma=0.01 ...............................................
    [CV] ...................................... C=100, gamma=0.01 -   0.0s
    [CV] C=100, gamma=0.01 ...............................................
    [CV] ...................................... C=100, gamma=0.01 -   0.0s
    [CV] C=100, gamma=0.01 ...............................................
    [CV] ...................................... C=100, gamma=0.01 -   0.0s
    [CV] C=100, gamma=0.001 ..............................................
    [CV] ..................................... C=100, gamma=0.001 -   0.0s
    [CV] C=100, gamma=0.001 ..............................................
    [CV] ..................................... C=100, gamma=0.001 -   0.0s
    [CV] C=100, gamma=0.001 ..............................................
    [CV] ..................................... C=100, gamma=0.001 -   0.0s
    

    [Parallel(n_jobs=1)]: Done  48 out of  48 | elapsed:    0.6s finished
    




    GridSearchCV(cv=None, error_score='raise',
           estimator=SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
      decision_function_shape='ovr', degree=3, gamma='auto', kernel='rbf',
      max_iter=-1, probability=False, random_state=None, shrinking=True,
      tol=0.001, verbose=False),
           fit_params={}, iid=True, n_jobs=1,
           param_grid={'C': [0.1, 1, 10, 100], 'gamma': [1, 0.1, 0.01, 0.001]},
           pre_dispatch='2*n_jobs', refit=True, scoring=None, verbose=2)



** Now take that grid model and create some predictions using the test set and create classification reports and confusion matrices for them. Were you able to improve?**


```python
grid_predictions  = grid_search.predict(X_test)
```


```python
confusion_matrix(y_test,grid_predictions)
```




    array([[19,  0,  0],
           [ 0, 13,  0],
           [ 0,  0, 13]], dtype=int64)




```python
print(classification_report(y_test, predictions))
```

                 precision    recall  f1-score   support
    
         setosa       1.00      1.00      1.00        19
     versicolor       1.00      1.00      1.00        13
      virginica       1.00      1.00      1.00        13
    
    avg / total       1.00      1.00      1.00        45
    
    

You should have done about the same or exactly the same, this makes sense, there is basically just one point that is too noisey to grab, which makes sense, we don't want to have an overfit model that would be able to grab that.

## Great Job!
