# Databricks notebook source


# COMMAND ----------

# MAGIC %sql
# MAGIC --Task 1 SQL Manipulation in Queries Tab:
# MAGIC USE  hackathon.team_4;
# MAGIC
# MAGIC CREATE OR REPLACE TABLE event_type_lookup (
# MAGIC   event_id INT,
# MAGIC   event_name STRING
# MAGIC );
# MAGIC
# MAGIC INSERT INTO event_type_lookup
# MAGIC VALUES
# MAGIC   (1, 'Shot Made'),
# MAGIC   (2, 'Shot Missed'),
# MAGIC   (3, 'Free Throw'),
# MAGIC   (4, 'Rebound'),
# MAGIC   (5, 'Turnover'),
# MAGIC   (6, 'Foul'),
# MAGIC   (7, 'Violation'),
# MAGIC   (8, 'Substitution'),
# MAGIC   (9, 'Timeout'),
# MAGIC   (10, 'Jump ball'),
# MAGIC   (12, 'Start of quarter'),
# MAGIC   (13, 'End of quarter');

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Task 2A SQL Manipulation in Data Tab of Lakehouse Dashboard:
# MAGIC -- Season_Events_Counts
# MAGIC
# MAGIC USE hackathon.team_4;
# MAGIC
# MAGIC CREATE OR REPLACE TABLE Player_Season_Event_Counts AS
# MAGIC SELECT
# MAGIC   P.season,
# MAGIC   ET.event_name AS event_type,
# MAGIC   P.count
# MAGIC FROM hackathon.team_4.play_by_event_gold P
# MAGIC JOIN hackathon.team_4.event_type_lookup ET
# MAGIC ON P.event_type = ET.event_id
# MAGIC ORDER BY P.season DESC;
# MAGIC
# MAGIC
# MAGIC CREATE OR REPLACE TABLE Player_Season_Event_Counts AS
# MAGIC SELECT
# MAGIC   P.season,
# MAGIC   ET.event_name AS event_type,
# MAGIC   P.player1_name,
# MAGIC   P.count
# MAGIC FROM hackathon.example_schema.play_by_player_per_season_gold P
# MAGIC JOIN hackathon.team_4.event_type_lookup ET
# MAGIC ON P.event_type = ET.event_id
# MAGIC ORDER BY P.season DESC;
# MAGIC
# MAGIC

# COMMAND ----------

# MAGIC %sql
# MAGIC --Task 2B SQL Manipulation in Data Tab of Lakehouse Dashboard:
# MAGIC USE hackathon.team_4;
# MAGIC CREATE OR REPLACE TABLE team_stats_non_nba AS
# MAGIC
# MAGIC
# MAGIC select h.home_team_name as team_name,h.season,
# MAGIC h.count as count_home,
# MAGIC a.count as count_away,
# MAGIC h.avg_points_for as avg_points_for_home,
# MAGIC a.avg_points_for as avg_points_for_away,
# MAGIC round(h.avg_points_for-a.avg_points_for,0) as margin,
# MAGIC h.avg_points_against as avg_points_against_home,
# MAGIC a.avg_points_against as avg_points_against_away
# MAGIC
# MAGIC from 
# MAGIC hackathon.team_4.team_stats_season_home_gold h
# MAGIC
# MAGIC join hackathon.team_4.team_stats_season_away_gold a
# MAGIC
# MAGIC on 
# MAGIC h.home_team_name=a.away_team_name and h.season=a.season
# MAGIC
# MAGIC where h.home_team_name not in ("New Jersey_Nets","LA_Clippers","Sacramento_Kings","Washington_Wizards","Indiana_Pacers","Miami_Heat","Milwaukee_Bucks","Denver_Nuggets","Philadelphia_76ers","Portland_Trail Blazers","Orlando_Magic","Los Angeles_Lakers","Toronto_Raptors","Memphis_Grizzlies","Cleveland_Cavaliers","Charlotte_Hornets","Atlanta_Hawks","Charlotte_Bobcats","Minnesota_Timberwolves","New Orleans/Oklahoma City_Hornets","Chicago_Bulls","Brooklyn_Nets","New York_Knicks","Oklahoma City_Thunder","Los Angeles_Clippers","San Antonio_Spurs","New Orleans_Hornets","Golden State_Warriors","Detroit_Pistons","Dallas_Mavericks","Phoenix_Suns","New Orleans_Pelicans","Boston_Celtics","Utah_Jazz","Houston_Rockets","Vancouver_Grizzlies","Seattle_SuperSonics","Washington_Bullets")
# MAGIC

# COMMAND ----------



# COMMAND ----------

# MAGIC %sql
# MAGIC
# MAGIC --Visual 1: Shot Made by Players over Seasons
# MAGIC
# MAGIC
# MAGIC select t1.season,t1.player1_name as player_name, t2.event_name, t1.count as count
# MAGIC from hackathon.team_4.play_by_player_gold t1
# MAGIC left join hackathon.team_4.event_type_lookup t2
# MAGIC on t1.event_type=t2.event_id
# MAGIC where t1.player1_name is not NULL  and t2.event_name is not null and t2.event_name = 'Shot Made' 
# MAGIC and t1.player1_name in ('Aaron Gordon','Aaron Holiday', 'Admiral Schofield')
# MAGIC
# MAGIC order by 3,4 desc
# MAGIC
# MAGIC
# MAGIC
# MAGIC
# MAGIC
# MAGIC
# MAGIC

# COMMAND ----------

# MAGIC %sql
# MAGIC --Visual 2: Event Count over seasons by Team (Top 5) - NBA
# MAGIC SELECT
# MAGIC   P.season,
# MAGIC   ET.event_name AS event_type,
# MAGIC   P.count
# MAGIC FROM hackathon.team_4.play_by_event_gold P
# MAGIC JOIN hackathon.team_4.event_type_lookup ET
# MAGIC ON P.event_type = ET.event_id
# MAGIC ORDER BY P.season DESC;
# MAGIC

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Visual 3: Event Count for each season by player (top 5)
# MAGIC WITH RankedPlayers AS (
# MAGIC   SELECT
# MAGIC     P.season,
# MAGIC     ET.event_name AS event_type,
# MAGIC     P.player1_name,
# MAGIC     P.count,
# MAGIC     RANK() OVER (PARTITION BY P.season ORDER BY P.count DESC) AS ranking
# MAGIC   FROM hackathon.example_schema.play_by_player_per_season_gold P
# MAGIC   JOIN hackathon.team_4.event_type_lookup ET
# MAGIC   ON P.event_type = ET.event_id
# MAGIC   WHERE P.player1_name IS NOT NULL
# MAGIC     AND P.season IN (
# MAGIC       SELECT DISTINCT season
# MAGIC       FROM hackathon.example_schema.play_by_player_per_season_gold
# MAGIC       ORDER BY season DESC
# MAGIC       LIMIT 10
# MAGIC     )
# MAGIC )
# MAGIC SELECT
# MAGIC   season,
# MAGIC   event_type,
# MAGIC   player1_name,
# MAGIC   count
# MAGIC FROM RankedPlayers
# MAGIC WHERE ranking <= 3;

# COMMAND ----------

# MAGIC %sql
# MAGIC --Visual 4: Avg points per season home and away
# MAGIC
# MAGIC select season,team_name, avg_points_for_home,avg_points_for_away
# MAGIC from
# MAGIC hackathon.team_4.team_stats_non_nba
# MAGIC

# COMMAND ----------

# #Machine Learning
# sdf = spark.sql("SELECT * FROM hackathon.team_4.team_stats_non_nba")
# sdf.head(10)

# COMMAND ----------

# df = sdf.toPandas()
# df = df.drop(["avg_points_against_home", "avg_points_against_away"], axis=1)


# COMMAND ----------

# def extract_last_two_digits(year_string):
#     return int(year_string[-2:])

# df['season_two_digits'] = df['season'].apply(extract_last_two_digits)
# df = df.drop(["season"], axis=1)
# df.head(5)

# COMMAND ----------

# #

# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LinearRegression
# from sklearn.metrics import mean_squared_error



# # Define the features (X) and target variable (y)
# X = df[['count_home', 'count_away', 'avg_points_for_home', 'avg_points_for_away', 'season_two_digits']]
# y = df['margin']

# # Split the data into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Create a linear regression model
# model = LinearRegression()

# # Fit the model to the training data
# model.fit(X_train, y_train)

# # Make predictions on the testing data
# y_pred = model.predict(X_test)

# # Calculate the Mean Squared Error to evaluate the model's performance
# mse = mean_squared_error(y_test, y_pred)
# print(f"Mean Squared Error: {mse}")

# # Now, you can use the trained model to make predictions for new data
# new_data = pd.DataFrame({'count_home': [44],
#                          'count_away': [44],
#                          'avg_points_for_home': [98],
#                          'avg_points_for_away': [98],
#                          'season_two_digits': [24]})
# new_predictions = model.predict(new_data)
# print("Predictions for new data:")
# print(new_predictions)


# COMMAND ----------

