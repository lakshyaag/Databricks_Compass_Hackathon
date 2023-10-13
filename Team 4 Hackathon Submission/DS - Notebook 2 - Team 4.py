# Databricks notebook source
# MAGIC %pip install databricks-feature-store databricks

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

from pyspark.sql.functions import pandas_udf, PandasUDFType
import pyspark.pandas as ps
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from databricks.feature_store import feature_table, FeatureStoreClient

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.model_selection import cross_validate, GridSearchCV
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.pipeline import Pipeline



# COMMAND ----------

# MAGIC %md
# MAGIC ## Load data from Feature Store

# COMMAND ----------

fs = FeatureStoreClient()
feature_set = fs.read_table(name="hackathon.example_schema.ml_nba_game_features_small")
display(feature_set)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Split into Training and Testing Sets Based on Season and Period and convert to pandas

# COMMAND ----------

# training set include any number of seasons or any data not in 4th quarter
# testing set only previous season 4th quarter (2022-23)
# drop game_id and event_num among other columns as these are the identifiers
# keep an original copy to be able to visualize actuals and predictions and filter properly for a game at the end of the notebook

# Maintain an original copy
feature_set_og = feature_set

# Drop specified columns and categorical ones
df = feature_set.drop("game_id", "event_num","home_team","team_leading_complete")

# Filter out 4th quarter data
df_no_4th_quarter = df.filter((df.period != 4)&(df.year!='2022'))

# Training data: Any number of seasons or any data not in the 4th quarter
train = df_no_4th_quarter.toPandas()
train_orig = df_no_4th_quarter.toPandas()

# Testing data: Only previous season 4th quarter (2022-23)
test = df.filter((df.period == 4) & (df.year == "2022")).toPandas()
test_orig = df.filter((df.period == 4) & (df.year == "2022")).toPandas()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Implement Modelling
# MAGIC * Test out a bunch of different regression models

# COMMAND ----------

from pyspark.sql.functions import col

# y_train will be a DataFrame consisting of the 'score_margin_complete' column
y_train = train['score_margin_complete']

# X_train will be the train DataFrame but without the 'score_margin_complete' column
X_train = train.drop('score_margin_complete',axis=1)

y_test = test['score_margin_complete']
X_test = test.drop('score_margin_complete',axis=1)

# COMMAND ----------

from sklearn.model_selection import cross_val_score

models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree Regression": DecisionTreeRegressor()
    
}

cv = 5  # Number of cross validation folds
scoring = 'neg_mean_absolute_error'  # Scoring metric

all_metrics = {}

# Train and compute metrics for each model
for name, model in models.items():
    scores = cross_val_score(model, X_train, y_train, cv=cv, scoring=scoring)  
   
    # Scores returned by cross_val_score are negative as it is a loss function. 
    # The actual MAE is simply the positive version of the score.
    mae_scores = -scores
    
    # Calculate mean and standard deviation of MAE
    mae_mean = mae_scores.mean()
    mae_std = mae_scores.std()
    
    all_metrics[name] = {
        "Mean of MAE": mae_mean,
        "Std of MAE": mae_std
    }

# Print metrics
for model, metrics in all_metrics.items():
    print(f"{model}:\nMean of MAE: {metrics['Mean of MAE']}\nStd of MAE: {metrics['Std of MAE']}\n")

# COMMAND ----------

# MAGIC %md
# MAGIC ### **OPTIONAL** Hyperparameter tune best model generated on baseline using all data 

# COMMAND ----------

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import make_scorer, r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeRegressor

pipeline = Pipeline(steps=[("scaler", StandardScaler()), ("dt", DecisionTreeRegressor())])

param_grid = {
    "dt__max_depth": [2, 5, 10],
    "dt__min_samples_split": [2, 5],
    "dt__min_samples_leaf": [1, 2],
}

scorers = {
    'r2_score': make_scorer(r2_score),
    'mean_squared_error': make_scorer(mean_squared_error),
    'mean_absolute_error': make_scorer(mean_absolute_error)
}

grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring=scorers, refit='mean_absolute_error', verbose=3)

grid_search.fit(X_train, y_train)

best_parameters = grid_search.best_params_

best_score = grid_search.best_score_

print("Best Parameters: ", best_parameters)
print("Best Score: ", best_score)

# COMMAND ----------

best_model = grid_search.best_estimator_

# COMMAND ----------

# MAGIC %md
# MAGIC ### Evaluate Results on Testing Set and Make Predictions

# COMMAND ----------

# Evaluate the MAE of your model on the test set you held out originally
y_preds = best_model.predict(X_test)

print(f"R2 Score on Test Set: {round(r2_score(y_test, y_preds), 3)}")
print(f"MSE on Test Set: {round(mean_squared_error(y_test, y_preds), 3)}")
print(f"MAE Score on Test Set: {round(mean_absolute_error(y_test, y_preds), 3)}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Visualize Predictions with Actuals for a Few Games

# COMMAND ----------

train_orig = train_orig
test_orig = test_orig

# COMMAND ----------

# recombine datasets
train_orig['predictions'] = best_model.predict(X_train) 
test_orig['predictions'] = best_model.predict(X_test)


# COMMAND ----------


plt.plot(df['minute_game'], df['score_margin_complete'], label="score_margin_complete")
plt.plot(df['minute_game'], df['predictions'], label="predictions")

plt.show()

# COMMAND ----------

display(test_orig)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Save Results to Catalog

# COMMAND ----------

feature_spark = spark.createDataFrame(feature_preds[['game_id', 'event_num', 'minute_game', 'score_margin_complete', 'predictions']])
feature_spark.write.mode("overwrite").saveAsTable("hackathon.example_schema.ml_results")

# COMMAND ----------

