Convert from `float64` to `float32` precision which is fine for most tasks

```py
prop = pd.read_csv('../input/properties_2016.csv')
convert = prop.dtypes == 'float64'
prop.loc[:, convert] = \
    prop.loc[:, convert].apply(lambda x: x.astype(np.float32))
```

Source - https://www.kaggle.com/flennerhag/ml-ensemble-scikit-learn-style-ensemble-learning/code
<!--stackedit_data:
eyJoaXN0b3J5IjpbLTE5NjgxMjA3NjJdfQ==
-->