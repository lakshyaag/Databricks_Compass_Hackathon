# Databricks notebook source
# MAGIC %pip install databricks-feature-store

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

from databricks.feature_store import feature_table, FeatureStoreClient
import pyspark.pandas as ps
ps.set_option('compute.ops_on_diff_frames', True)
from pyspark.ml.feature import MinMaxScaler
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# COMMAND ----------

# replace with your schema if you complete data engineering Silver Table Tasks
nba_df = spark.table("hackathon.team_4.play_by_play_silver")
display(nba_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Perform Some EDA
# MAGIC * Check datatypes, missing values, remove columns and clean data

# COMMAND ----------

# convert dataframe to be pandas compatible
df = nba_df.pandas_api()

# COMMAND ----------

# Check for data quality here
display(df.info())

# COMMAND ----------

display(df.isnull().sum())

# COMMAND ----------

df = df.drop_duplicates()

# COMMAND ----------

# remove columns with lots of empty values, duplicate identifiers, uninformative/unusable columns, duplicate information
columns_to_drop = [
    'event_action_type',
    'minute_remaining_quarter',
    'seconds_remaining_quarter',
    'neutral_description',
    'play_description',
    'score',
    'home_score',
    'away_score',
    'score_margin',
    'team_leading',
    'player1_team_nickname',
    'player1_team_city',
    'player2_team_nickname',
    'player2_team_city',
    'player3_team_nickname',
    'player3_team_city',   
    'wc_time_string',
    'time_quarter',
    'person1type', 
    'player1_id',
    'person2type',
    'player2_id', 'person3type', 'player3_id', 'video_available_flag', 'player1_name', 'player1_team_id', 'player1_team_abbreviation', 
    'player2_name', 'player2_team_id', 'player2_team_abbreviation', 'player3_name', 'player3_team_id', 'player3_team_abbreviation'
]

df = df.drop(columns_to_drop, axis=1)

# COMMAND ----------

# filter for only regulation time
df = df.loc[df['period'] <= 4]

# COMMAND ----------

# remove non-official nba teams use data dictionary for list
team_list = [
    "AtlantaHawks",
    "BostonCeltics",
    "BrooklynNets",
    "CharlotteBobcats",
    "CharlotteHornets",
    "ChicagoBulls",
    "ClevelandCavaliers",
    "DallasMavericks",
    "DenverNuggets",
    "DetroitPistons",
    "Golden StateWarriors",
    "HoustonRockets",
    "IndianaPacers",
    "LAClippers",
    "Los AngelesClippers",
    "Los AngelesLakers",
    "MemphisGrizzlies",
    "MiamiHeat",
    "MilwaukeeBucks",
    "MinnesotaTimberwolves",
    "New JerseyNets",
    "New Orleans/Oklahoma CityHornets",
    "New OrleansHornets",
    "New OrleansPelicans",
    "New YorkKnicks",
    "Oklahoma CityThunder",
    "OrlandoMagic",
    "Philadelphia76ers",
    "PhoenixSuns",
    "PortlandTrail Blazers",
    "SacramentoKings",
    "San AntonioSpurs",
    "SeattleSuperSonics",
    "TorontoRaptors",
    "UtahJazz",
    "VancouverGrizzlies",
    "WashingtonBullets",
    "WashingtonWizards",
]

# Keep only rows where either home or away team is in the teamlist
df = df[(df['home_team'].isin(team_list)) & (df['away_team'].isin(team_list))]

# COMMAND ----------

# rename old team names using data dictionary
rename_team_names = {
    "Charlotte_Bobcats": "Charlotte_Hornets",
    "LA_Clippers": "Los Angeles_Clippers",
    "Vancouver_Grizzlies": "Memphis_Grizzlies",
    "Seattle_SuperSonics": "Oklahoma City_Thunder",
    "New Orleans/Oklahoma City_Hornets": "New Orleans_Pelicans",
    "New Orleans_Hornets": "New Orleans_Pelicans",
    "New Jersey_Nets": "Brooklyn_Nets",
    "Washington_Bullets": "Washington_Wizards"
}

df['home_team'] = df['home_team'].replace(rename_team_names)
df['away_team'] = df['away_team'].replace(rename_team_names)


# COMMAND ----------

display(df)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Visualize a Game Dataset

# COMMAND ----------

# MAGIC %md
# MAGIC ### Visualize multiple games

# COMMAND ----------

games = df['game_id'].unique().tolist()[:20]

# COMMAND ----------

df_game = df[df['game_id'].isin(games)].to_pandas()

# COMMAND ----------

df_game.shape

# COMMAND ----------

# create a plot of 20 games
fig, ax = plt.subplots(figsize=(15, 10))

sns.lineplot(x='minute_game', y='score_margin_complete', hue='game_id', data=df_game, ax=ax)

plt.title('Score Margin Over Time by Game', fontsize=20)

plt.xlabel('Minute of Game', fontsize=15)
plt.ylabel('Score Margin', fontsize=15)

plt.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=12)
plt.show()

# COMMAND ----------

fig, ax = plt.subplots(ncols=2, figsize=(15, 10))

ax[0] = sns.countplot(
    y="home_team",
    data=df_game,
    
    order=df_game["home_team"].value_counts().index,
    color="blue",
    ax=ax[0],
)
ax[0].set_title("Number of Games by Home Team")
ax[0].set_xlabel("Home Team")
ax[0].set_ylabel("Number of Games")

ax[1] = sns.countplot(
    y="away_team",
    data=df_game,
    order=df_game["away_team"].value_counts().index,
    color="red",
    ax=ax[1],
)
ax[1].set_title("Number of Games by Away Team")
ax[1].set_xlabel("Away Team")
ax[1].set_ylabel("Number of Games")

plt.suptitle("Number of Games by Team", fontsize=20)
plt.tight_layout()
plt.show()

# COMMAND ----------

display(df_game)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Check Numerical Univariate Distributions and Correlations

# COMMAND ----------

# select numerical columns and check distributions and correlations
numeric_columns = [
    'event_type',
    'period',
    'minute_game',
    'score_margin_complete',
    'away_score_complete',
    'home_score_complete'
]

display(df_game[numeric_columns].describe())

# COMMAND ----------

# Plot univariate distributions of numeric columns
fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(15, 10))

for i, column in enumerate(numeric_columns):
    sns.histplot(df_game, x=column, ax=ax[i//3][i%3])

plt.tight_layout()
plt.show()


# COMMAND ----------

sns.pairplot(df_game[numeric_columns])

# COMMAND ----------

plt.figure(figsize=(12, 10))

corr = df_game.corr()
sns.heatmap(
    corr,
    mask=np.triu(np.ones_like(corr, dtype=bool)),
    cmap=sns.diverging_palette(220, 10, as_cmap=True, center="light"),
    vmax=0.3,
    center=0,
    square=True,
    annot=True,
    fmt=".1f",
    linewidths=0.5,
    cbar_kws={"shrink": 0.5},
)
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Prepare Data for forecasting 

# COMMAND ----------

# MAGIC %md
# MAGIC #### Create Seasonality Variables from game date

# COMMAND ----------

# month and day from game_date and year from season
df['game_date'] = ps.to_datetime(df['game_date'])
df['game_month'] = df['game_date'].dt.month
df['game_day'] = df['game_date'].dt.day

df['game_year'] = df['season'].apply(lambda x: x.split('-')[0]).astype(int)

# COMMAND ----------

display(df)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Create within game lagging metrics

# COMMAND ----------

### Within game lagging metric
# calculate momentum within a game by taking the difference between the margin 40 events which is just less than the last four minutes of play
# 455 plays per game 455 / 48 is roughly 10 making 40 about 4 minutes
# Hint Use a GroupBy and shift
# remove rows with missing values using dropna and subset

df["score_margin_40_events_ago"] = df.groupby('game_id')['score_margin_complete'].shift(40)

df['momentum_40'] = df['score_margin_complete'] - df['score_margin_40_events_ago']

df = df.dropna(subset=['momentum_40'])

# COMMAND ----------

# MAGIC %md
# MAGIC #### Create small and larger dataset by subsetting for only Toronto Raptors and One-Hot Encode Categorical Columns

# COMMAND ----------

df_small = df[df['home_team'] == "Toronto_Raptors"]

# rename columns for storing in feature store
df_small.columns = map(lambda x: str(x).replace(" ", "_").lower(), df_small.columns)

# COMMAND ----------

df_small = ps.get_dummies(df_small, columns=['event_type', 'away_team'])

# COMMAND ----------

display(df_small.info())

# COMMAND ----------

# MAGIC %md
# MAGIC ### Write to Feature Store

# COMMAND ----------

spark_df_small = df_small.to_spark()

# COMMAND ----------

# replace example_schema with your schema
table_name = "hackathon.team_4.ml_nba_game_features_small"
fs = FeatureStoreClient()

# write code to write to feature store below
fs.write_table(
    table_name,
    df = spark_df_small,
    mode="overwrite"
    
)