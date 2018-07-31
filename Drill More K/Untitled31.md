# Start

This part is copied from the lesson.

```python
import numpy as np
import pandas as pd
import scipy
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA
from sklearn.cluster import MiniBatchKMeans
%matplotlib inline
```


```python
df = pd.read_csv("https://raw.githubusercontent.com/Thinkful-Ed/data-201-resources/master/cleveland.csv", header=None, error_bad_lines=False)

# Define the features and the outcome.
X = df.iloc[:, :13]
y = df.iloc[:, 13]

# Replace missing values (marked by ?) with a 0.
X = X.replace(to_replace='?', value=0)

# Binarize y so that 1 means heart disease diagnosis and 0 means no diagnosis.
y = np.where(y > 0, 0, 1)
```


```python
# Normalize the data.
X_norm = normalize(X)

# Reduce it to two components.
X_pca = PCA(2).fit_transform(X_norm)

# Calculate predicted values.
y_pred = KMeans(n_clusters=2, random_state=42).fit_predict(X_pca)

# Plot the solution.
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y_pred)
plt.show()

# Check the solution against the data.
print('Comparing k-means clusters against the data:')
print(pd.crosstab(y_pred, y))
```


![png](output_2_0.png)


    Comparing k-means clusters against the data:
    col_0   0   1
    row_0        
    0      84  65
    1      55  99
    

So that looks like 183 correct?

```python
# Each batch will be made up of 200 data points.
minibatchkmeans = MiniBatchKMeans(
    init='random',
    n_clusters=2,
    batch_size=200)
minibatchkmeans.fit(X_pca)

# Add the new predicted cluster memberships to the data frame.
predict_mini = minibatchkmeans.predict(X_pca)

# Check the MiniBatch model against our earlier one.
print('Comparing k-means and mini batch k-means solutions:')
print(pd.crosstab(predict_mini, y_pred))
```

    Comparing k-means and mini batch k-means solutions:
    col_0    0    1
    row_0          
    0      149    5
    1        0  149
    
This is much worse.  Why would we use minibatchkmeans?

What I do below is to test all the k sizes between 0 and 9 to see if they get more accurate.

```python
pred = list()
accuracy = list()
best = 0
it = 0
for i in range(9):
    k = i + 2
    X_pca = PCA(k).fit_transform(X_norm)

    y_pred = KMeans(n_clusters=k).fit_predict(X_pca)


    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y_pred)
    plt.show()

    print('Comparing k-means clusters against the data:')
    conf = pd.crosstab(y_pred, y)
    pred.append(conf)
    this_acc = (conf[1][0] + conf[0][1])**2
    accuracy.append(this_acc)
    if (this_acc > best):
        best = this_acc
        it = i
    print(str(i) + " " + str(this_acc))

print(":: " + str(it) + " " + str(best))
print(pred[it])
    
```


![png](output_5_0.png)


    Comparing k-means clusters against the data:
    0 33489
    


![png](output_5_2.png)


    Comparing k-means clusters against the data:
    1 9409
    


![png](output_5_4.png)


    Comparing k-means clusters against the data:
    2 4624
    


![png](output_5_6.png)


    Comparing k-means clusters against the data:
    3 9604
    


![png](output_5_8.png)


    Comparing k-means clusters against the data:
    4 1296
    


![png](output_5_10.png)


    Comparing k-means clusters against the data:
    5 1681
    


![png](output_5_12.png)


    Comparing k-means clusters against the data:
    6 5041
    


![png](output_5_14.png)


    Comparing k-means clusters against the data:
    7 529
    


![png](output_5_16.png)


    Comparing k-means clusters against the data:
    8 1936
    :: 0 33489
    col_0   0   1
    row_0        
    0      55  99
    1      84  65
    

The answer is disappointing in that the k as 2 so 50/50 is best.  We can see the accuracy of differing k below.

```python
plt.plot(accuracy)
```




    [<matplotlib.lines.Line2D at 0x141cfcd0>]




![png](output_6_1.png)


Now we try something different.  We will print out all the different PCA and k sizes, both between 10 and 10 but with the same value.

```python
pred2 = list()
accuracy2 = list()
best2 = 0
it2 = 0
for i in range(9):
    k = i + 2
    X_pca = PCA(k).fit_transform(X_norm)

    y_pred = KMeans(n_clusters=k).fit_predict(X_pca)


    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y_pred)
    plt.show()

    print('Comparing k-means clusters against the data:')
    conf = pd.crosstab(y_pred, y)
    pred2.append(conf)
    this_acc2 = (conf[1][0] + conf[0][1])**2
    accuracy2.append(this_acc2)
    if (this_acc2 > best2):
        best2 = this_acc2
        it2 = i
    print(str(i) + " " + str(this_acc2))

print(":: " + str(it2) + " " + str(best2))
print(pred2[it2])
    
```


![png](output_7_0.png)


    Comparing k-means clusters against the data:
    0 33489
    


![png](output_7_2.png)


    Comparing k-means clusters against the data:
    1 5625
    


![png](output_7_4.png)


    Comparing k-means clusters against the data:
    2 11236
    


![png](output_7_6.png)


    Comparing k-means clusters against the data:
    3 1764
    


![png](output_7_8.png)


    Comparing k-means clusters against the data:
    4 2809
    


![png](output_7_10.png)


    Comparing k-means clusters against the data:
    5 2704
    


![png](output_7_12.png)


    Comparing k-means clusters against the data:
    6 1521
    


![png](output_7_14.png)


    Comparing k-means clusters against the data:
    7 484
    


![png](output_7_16.png)


    Comparing k-means clusters against the data:
    8 1225
    :: 0 33489
    col_0   0   1
    row_0        
    0      55  99
    1      84  65
    

Very disappointing.  Still 50/50 is best.  Checking the accuracy below we see a very identifical chart of dropping accuracy with greater k.  Why did thinkful give us this data set if it doesn't show the usefulness of the algorithm at different settings?

```python
plt.plot(accuracy2)
```




    [<matplotlib.lines.Line2D at 0x181fed0>]




![png](output_8_1.png)

Now I try the final combination of all different k and PCA sizes, so that complexity time is n^2.  Let's see if some combination of this allows us different accuracy.


```python
pred3 = list()
accuracy3 = list()
best3 = 0
it3 = 0
jt3 = 0
for i in range(9):
    k = i + 2
    pred3.append(list())
    accuracy3.append(list())
    for j in range(9):
        h = j + 2
        X_pca = PCA(h).fit_transform(X_norm)

        y_pred = KMeans(n_clusters=k).fit_predict(X_pca)

        print('Comparing k-means clusters against the data:')
        conf = pd.crosstab(y_pred, y)
        pred3[i].append(conf)
        this_acc3 = (conf[1][0] + conf[0][1])**2
        accuracy3[i].append(this_acc3)
        if (this_acc3 > best3):
            best3 = this_acc3
            it3 = i
            jt3 = j
        print(str(i) + " " + str(h) + " " + str(this_acc3))

print(":: " + str(it3) + " " + " " + str(jt3) + " " + str(best3))
print(pred3[it3][jt3])
    
```


Comparing k-means clusters against the data:
0 2 14400
Comparing k-means clusters against the data:
0 3 33489
Comparing k-means clusters against the data:
0 4 14400
Comparing k-means clusters against the data:
0 5 33489
Comparing k-means clusters against the data:
0 6 14400
Comparing k-means clusters against the data:
0 7 14400
Comparing k-means clusters against the data:
0 8 33489
Comparing k-means clusters against the data:
0 9 14400
Comparing k-means clusters against the data:
0 10 14641
Comparing k-means clusters against the data:
1 2 14884
Comparing k-means clusters against the data:
1 3 11881
Comparing k-means clusters against the data:
1 4 3364
Comparing k-means clusters against the data:
1 5 9604
Comparing k-means clusters against the data:
1 6 9409
Comparing k-means clusters against the data:
1 7 3481
Comparing k-means clusters against the data:
1 8 16129
Comparing k-means clusters against the data:
1 9 12100
Comparing k-means clusters against the data:
1 10 3481
Comparing k-means clusters against the data:
2 2 8649
Comparing k-means clusters against the data:
2 3 11025
Comparing k-means clusters against the data:
2 4 9604
Comparing k-means clusters against the data:
2 5 9216
Comparing k-means clusters against the data:
2 6 3721
Comparing k-means clusters against the data:
2 7 4096
Comparing k-means clusters against the data:
2 8 1444
Comparing k-means clusters against the data:
2 9 10609
Comparing k-means clusters against the data:
2 10 4489
Comparing k-means clusters against the data:
3 2 1849
Comparing k-means clusters against the data:
3 3 1849
Comparing k-means clusters against the data:
3 4 3844
Comparing k-means clusters against the data:
3 5 3025
Comparing k-means clusters against the data:
3 6 8464
Comparing k-means clusters against the data:
3 7 3600
Comparing k-means clusters against the data:
3 8 3600
Comparing k-means clusters against the data:
3 9 8836
Comparing k-means clusters against the data:
3 10 8836
Comparing k-means clusters against the data:
4 2 1600
Comparing k-means clusters against the data:
4 3 1156
Comparing k-means clusters against the data:
4 4 1764
Comparing k-means clusters against the data:
4 5 324
Comparing k-means clusters against the data:
4 6 841
Comparing k-means clusters against the data:
4 7 4356
Comparing k-means clusters against the data:
4 8 576
Comparing k-means clusters against the data:
4 9 1225
Comparing k-means clusters against the data:
4 10 961
Comparing k-means clusters against the data:
5 2 5476
Comparing k-means clusters against the data:
5 3 484
Comparing k-means clusters against the data:
5 4 961
Comparing k-means clusters against the data:
5 5 4096
Comparing k-means clusters against the data:
5 6 2304
Comparing k-means clusters against the data:
5 7 4356
Comparing k-means clusters against the data:
5 8 1024
Comparing k-means clusters against the data:
5 9 6724
Comparing k-means clusters against the data:
5 10 4489
Comparing k-means clusters against the data:
6 2 5476
Comparing k-means clusters against the data:
6 3 4624
Comparing k-means clusters against the data:
6 4 2304
Comparing k-means clusters against the data:
6 5 2209
Comparing k-means clusters against the data:
6 6 1225
Comparing k-means clusters against the data:
6 7 1600
Comparing k-means clusters against the data:
6 8 2809
Comparing k-means clusters against the data:
6 9 1849
Comparing k-means clusters against the data:
6 10 961
Comparing k-means clusters against the data:
7 2 2809
Comparing k-means clusters against the data:
7 3 1764
Comparing k-means clusters against the data:
7 4 1369
Comparing k-means clusters against the data:
7 5 676
Comparing k-means clusters against the data:
7 6 484
Comparing k-means clusters against the data:
7 7 2916
Comparing k-means clusters against the data:
7 8 2401
Comparing k-means clusters against the data:
7 9 1600
Comparing k-means clusters against the data:
7 10 1024
Comparing k-means clusters against the data:
8 2 121
Comparing k-means clusters against the data:
8 3 729
Comparing k-means clusters against the data:
8 4 1600
Comparing k-means clusters against the data:
8 5 1681
Comparing k-means clusters against the data:
8 6 2401
Comparing k-means clusters against the data:
8 7 2116
Comparing k-means clusters against the data:
8 8 1521
Comparing k-means clusters against the data:
8 9 529
Comparing k-means clusters against the data:
8 10 2809
:: 0  1 33489
col_0   0   1
row_0        
0      55  99
1      84  65
    
Now we see that 50/50 is indeed the best accuracy for this dataset.  I wonder why.  Is it because it is too small?  The graph below shows the falling accuracy is the same as the accuracy charts above.

```python
plt.plot(accuracy3)
```




    [<matplotlib.lines.Line2D at 0x18d9310>,
     <matplotlib.lines.Line2D at 0x18d9a10>,
     <matplotlib.lines.Line2D at 0x18d9430>,
     <matplotlib.lines.Line2D at 0x18d9930>,
     <matplotlib.lines.Line2D at 0x18d9ad0>,
     <matplotlib.lines.Line2D at 0x18d9ff0>,
     <matplotlib.lines.Line2D at 0x18d9090>,
     <matplotlib.lines.Line2D at 0x18d9370>,
     <matplotlib.lines.Line2D at 0x18d9490>]




![png](output_10_1.png)

