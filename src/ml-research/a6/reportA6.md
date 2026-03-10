# Grid optimization

## RandomizedSearchCV 
RandomizedSearchCV to find around where the optimal could be, gave this result.

| Rank | Mean F1 (Weighted) | Std Dev | Kernel | C | Gamma | Degree | Coef0 |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | **0.5972** | 0.0328 | rbf | 10 | auto | 3 | 0.0 |
| 2 | 0.5928 | 0.0129 | poly | 10 | 0.01 | 3 | 1.0 |
| 3 | 0.5925 | 0.0235 | rbf | 10 | 0.01 | 3 | 1.0 |
| 4 | 0.5831 | 0.0230 | rbf | 100 | 0.01 | 3 | 0.0 |
| 5 | 0.5831 | 0.0230 | rbf | 100 | 0.01 | 2 | 1.0 |
| 6 | 0.5831 | 0.0230 | rbf | 100 | 0.01 | 2 | 0.0 |
| 7 | 0.5784 | 0.0225 | poly | 100 | scale | 2 | 1.0 |
| 8 | 0.5720 | 0.0450 | rbf | 100 | auto | 2 | 1.0 |
| 9 | 0.5585 | 0.0119 | linear | 100 | 0.01 | 3 | 0.0 |
| 10 | 0.5585 | 0.0119 | linear | 100 | scale | 2 | 0.0 |

## GridsearchCV

We then used GridSearchCV for poly and rbf aswell as narrowing the other values 

```python
refined_param_grid = {
    'svc__kernel': ['rbf', 'poly'],
    'svc__C': [5, 8, 10, 12, 15, 50, 75],
    'svc__gamma': ['auto', 0.01, 0.02], 
    'svc__degree': [3],                            
}
```


| Rank | Mean F1 (Weighted) | Std Dev | Kernel | C | Gamma | Coef0 | Degree |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **1** | **0.6026** | 0.0243 | rbf | 5 | 0.02 | 0.5 | 3 |
| **1** | **0.6026** | 0.0243 | rbf | 5 | 1.0 | 0.02 | 3 |
| **1** | **0.6026** | 0.0243 | rbf | 5 | 0.0 | 0.02 | 3 |
| 4 | 0.5978 | 0.0197 | rbf | 8 | 0.02 | 1.0 | 3 |
| 4 | 0.5978 | 0.0197 | rbf | 8 | 0.02 | 0.5 | 3 |
| 4 | 0.5978 | 0.0197 | rbf | 8 | 0.02 | 0.0 | 3 |
| 7 | 0.5972 | 0.0328 | rbf | 10 | auto | 0.0 | 3 |
| 7 | 0.5972 | 0.0328 | rbf | 10 | auto | 0.5 | 3 |
| 7 | 0.5972 | 0.0328 | rbf | 10 | auto | 1.0 | 3 |
| 10 | 0.5969 | 0.0270 | rbf | 8 | auto | 0.5 | 3 |

## Further improving

```python
refined_param_grid = {
    'svc__kernel': ['rbf'],
    'svc__C': [2, 3, 4, 5, 6],
    'svc__gamma': [0.015, 0.02, 0.025], 
}
```


| Rank | Mean F1 (Weighted) | Std Dev | Kernel | C | Gamma |
|:---:|:---:|:---:|:---:|:---:|:---:|
| **1** | **0.6026** | 0.0243 | rbf | 5 | 0.020 |
| 2 | 0.6007 | 0.0236 | rbf | 4 | 0.025 |
| 3 | 0.5983 | 0.0264 | rbf | 6 | 0.015 |
| 4 | 0.5947 | 0.0217 | rbf | 5 | 0.025 |
| 5 | 0.5939 | 0.0290 | rbf | 6 | 0.020 |
| 6 | 0.5937 | 0.0273 | rbf | 5 | 0.015 |
| 7 | 0.5928 | 0.0209 | rbf | 6 | 0.025 |
| 8 | 0.5927 | 0.0251 | rbf | 3 | 0.025 |
| 9 | 0.5926 | 0.0276 | rbf | 4 | 0.020 |
| 10 | 0.5891 | 0.0324 | rbf | 3 | 0.020 |

# Implementation into ensemble
When we found the optimized SVM we added that to the level0 models. This increased the F1 by a small bit but not significantly better.


# 5 Fold CV results
As we used 10 folds in our own comparisons a script was added to our utils diles that loaded the models @prod, @dev and @backup and did a new cross validation using 5 folds. This gave the following results.
## A2
R2 Mean: 0.7798
F1 Mean: 0.0102
## A3
F1 Mean: 0.6242
STD Mean: 0.0183
