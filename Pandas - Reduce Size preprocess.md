## Convert from `float64` to `float32` precision which is fine for most tasks

```py
prop = pd.read_csv('../input/properties_2016.csv')
convert = prop.dtypes == 'float64'
prop.loc[:, convert] = \
    prop.loc[:, convert].apply(lambda x: x.astype(np.float32))
```

Source - https://www.kaggle.com/flennerhag/ml-ensemble-scikit-learn-style-ensemble-learning/code

## Convert categorical to numeric using dictionary in pandas dataframe

```py
# Define a dictionary for the target mapping
target_map = {'Yes':1, 'No':0}
# Use the pandas apply method to numerically encode our attrition target variable
attrition["Attrition_numerical"] = attrition["Attrition"].apply(lambda x: target_map[x])
```
Source - https://www.kaggle.com/vikasg/predicting-ibm-attrition-acc-of-89-via-rf-gbm/edit


## Oversampling 
Since we have already noted the severe imbalance in the values within the target variable, let us implement the SMOTE method in the dealing with this skewed value via the imblearn Python package.
```py
from imblearn.over_sampling import SMOTE
oversampler=SMOTE(random_state=0)
smote_train, smote_target = oversampler.fit_sample(train,target_train)
```

## Initializing Random Forrest 

```py
seed = 0   
rf_params = {
    'n_jobs': -1,
    'n_estimators': 800,
    'warm_start': True, 
    'max_features': 0.3,
    'max_depth': 9,
    'min_samples_leaf': 2,
    'max_features' : 'sqrt',
    'random_state' : seed,
    'verbose': 0
}

rf = RandomForestClassifier(**rf_params)
```

```py
rf.fit(smote_train, smote_target)
print("Fitting of Random Forest as finished")
```

```py
rf_predictions = rf.predict(test)
print("Predictions finished")
```

## Supress warning messages

```py
import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'
```


## Run system commands 
```py
from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))
```

## Docker stats (Try it)

```sh
docker stats udacity_deep_learning
```

```
CONTAINER               CPU %               MEM USAGE / LIMIT     MEM %               NET I/O               BLOCK I/O
udacity_deep_learning   0.01%               1.402 GB / 12.88 GB   10.88%              256.3 MB / 657.9 kB   733.4 MB / 237.6 kB
```
<!--stackedit_data:
eyJoaXN0b3J5IjpbMTYzOTc3ODY0MSwtMTk2ODEyMDc2Ml19
-->