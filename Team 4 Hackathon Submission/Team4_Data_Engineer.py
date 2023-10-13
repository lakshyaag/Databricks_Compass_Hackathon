# Databricks notebook source
# MAGIC %md ## Import Necessary Libraries

# COMMAND ----------

import dlt
import pyspark.sql.functions as F
from pyspark.sql.window import Window
from pyspark.sql.functions import *

# COMMAND ----------

# MAGIC %md ## Read Raw NBA Games Data from S3 Bucket to Bronze Data

# COMMAND ----------

path = "s3://databricks-compass-hackathon-main-external-location/Montreal_Hackathon_2023/NBA_Games"
paths = dbutils.fs.ls(path)[-2:]

# COMMAND ----------

merged_df = dfs[0].unionByName(dfs[1])

# COMMAND ----------

display(merged_df)

# COMMAND ----------

display(df)

# COMMAND ----------

def nba_games_bronze(path):
    df = spark.read.format("parquet").option("inferSchema", "true").option("header", "true").load(path)
    return df

dfs = []
for p in paths:
    dfs.append(nba_games_bronze(p.path))

# COMMAND ----------



# COMMAND ----------

path = "s3://databricks-compass-hackathon-main-external-location/Montreal_Hackathon_2023/NBA_Games"
dbutils.fs.ls(path)

@dlt.table(
  name = "nba_games_bronze",
  comment = "Raw data ingested from S3 bucket" 
)
def nba_games_bronze():
    df = spark.read.format("parquet").option("inferSchema", "true").option("header", "true").load(path)
    return df

# COMMAND ----------

# MAGIC %md ### Clean Data and Ensure Data Quality Moving Data From Bronze to Silver

# COMMAND ----------

# MAGIC %md ### Create Two Tables From the Original Data
# MAGIC * Play-by-Play Data
# MAGIC * Overall Game Data

# COMMAND ----------

# MAGIC %md #### Play-by-Play Data

# COMMAND ----------

columns =["game_id","game_date","home_team_city",
"home_team_name",
"away_team_city",
"away_team_name",
"home_description",
"visitor_description",
"event_num",
"event_type",
"event_action_type",
"period",
"minute_game",
"wc_time_string",
"time_quarter",
"minute_remaining_quarter",
"seconds_remaining_quarter",
"neutral_description",
"person1type",
"player1_id","person2type",
"player2_id","person3type",
"player3_id",
"video_available_flag",
"player1_name",
"player1_team_id",
"player1_team_city",
"player1_team_nickname",
"player1_team_abbreviation",
"player2_name",
"player2_team_id",
"player2_team_city",
"player2_team_nickname",
"player2_team_abbreviation",
"player3_name",
"player3_team_id",
"player3_team_city",
"player3_team_nickname",
"player3_team_abbreviation",
"score",
"away_score",
"home_score",
"score_margin",
"team_leading"]

@dlt.create_table(
    name="Play_by_Play_silver",
    comment="Contains column specified in the Play_by_Play data dictionary, "
)
@dlt.expect("score_margin_complete does not contain any empty rows","score_margin_complete is not null")

def play_by_play_silver():
    window = (
        Window
        .partitionBy('game_id')
        .rowsBetween(Window.unboundedPreceding, Window.currentRow)
    )
    df = spark.table("LIVE.nba_games_bronze")
    df = df.select(*columns,df["`season.x`"].alias("season"))
    df = df.withColumn("score_margin",regexp_replace(col("score_margin"),"TIE","0"))
    df = df.withColumn("event_column",col("event_num").cast("integer").alias("event_num")).withColumn("event_type",col("event_type").cast("integer").alias("event_type")).withColumn(
        "event_action_type",col("event_action_type").cast("integer").alias("event_action_type")
    ).withColumn("video_available_flag",col("video_available_flag").cast("integer").alias("video_available_flag")).withColumn("score_margin",col("score_margin").cast("integer").alias("score_margin"))
    
    df = df.withColumn("home_team",concat(col("home_team_city"),col("home_team_name"))).drop("home_team_city","home_team_name").withColumn("away_team",concat(col("away_team_city"),col("away_team_name"))).drop("away_team_city","away_team_name").withColumn("play_description",concat(col("home_description"),col("visitor_description"))).drop("visitor_description","home_description")

    df = df.withColumn('score_margin', F.last('score_margin', ignorenulls=True).over(window)).withColumn('score', F.last('score', ignorenulls=True).over(window)).withColumn('away_score', F.last('away_score', ignorenulls=True).over(window)).withColumn('home_score', F.last('home_score', ignorenulls=True).over(window)).withColumn('team_leading', F.last('team_leading', ignorenulls=True).over(window))

    df = df.withColumn("score_margin_complete",col('score_margin')).withColumn("score_complete",col('score')).withColumn('away_score_complete',col('away_score')).withColumn('home_score_complete',col('home_score')).withColumn('team_leading_complete',col('team_leading'))

    df = df.fillna(0,subset=['score_margin_complete','away_score_complete','home_score_complete'])

    df = df.fillna('0 - 0',subset=['score_complete'])
    df = df.fillna('Home',subset=['team_leading_complete'])

    df = df.orderBy('game_id','event_num')
    df = df.dropDuplicates()

    return (df
                # apply Window function and data cleaning steps here
    )

# COMMAND ----------

# MAGIC %md #### Overall Game Data

# COMMAND ----------

@dlt.create_table(
name = "Overall_Game_Silver",
comment = "Game level data separated from raw data"
)
def overall_game_silver():
    # Define a reference to the bronze level table
    df_bronze = spark.table("LIVE.nba_games_bronze")

    # Define columns representing game level data
    game_level_columns = ["game_id", "game_date", "game_code", "day", "month_num", 
                          "week_number", "week_name", "if_necessary", "series_game_number",
                          "series_text", "arena_name", "arena_state", "arena_city",
                          "postponed_status", "branch_link", "game_subtype", "home_team_id",
                          "home_team_name", "home_team_city", "home_team_tricode", "home_team_slug",
                          "home_team_wins", "home_team_losses", "home_team_score", "home_team_seed",
                          "away_team_id", "away_team_name", "away_team_city", "away_team_tricode",
                          "away_team_slug", "away_team_wins", "away_team_losses", "away_team_score",
                          "away_team_seed" 
                          ]

    # Select game level columns from bronze table and drop duplicates
    df = df_bronze.select(*game_level_columns,df_bronze["`season.x`"].alias("season")).dropDuplicates()

    return df


# COMMAND ----------

# MAGIC %md ### Create Aggregated Data at the gold level

# COMMAND ----------

@dlt.create_table(
    comment="Aggregated play by event statistics by season and event type",
    name='Play_by_event_gold'
)
def play_by_event_gold():
    # Read the data from the silver table
    df = dlt.read("Play_by_Play_silver")

    # Perform groupBy and count operation
    df = df.groupBy(["season", "event_type"]).agg(F.count("*").alias("count"))

    return df

# COMMAND ----------

@dlt.create_table(
    comment="Aggregated play by event statistics by season, event type, and home team",
    name="Play_by_event_team_gold"
)
def play_by_event_team_gold():
    # Read the data from the silver table
    df = dlt.read("Play_by_Play_silver")

    # Perform groupBy and count operation
    df = df.groupBy(["season", "event_type", "home_team"]).agg(F.count("*").alias("count"))

    return df

# COMMAND ----------

@dlt.create_table(
    comment="Aggregated play by event statistics by season, event type, and player 1 name",
    name= 'Play_by_player_gold'
)
def play_by_player_gold():
    # Read the data from the silver table
    df = dlt.read("Play_by_Play_silver")

    # Perform groupBy and count operation
    df = df.groupBy(["season", "event_type", "player1_name"]).agg(F.count("*").alias("count"))

    return df

# COMMAND ----------

@dlt.create_table(
    comment="Aggregated statistics by home team and season"
)
def team_stats_season_home_gold():
    # Read the data from the silver table
    df = spark.table("LIVE.Overall_Game_Silver")

    # Perform groupBy and aggregation operations
    df = (df.groupBy(["home_team_name", "season"])
            .agg(F.count("*").alias("count"),
                 F.avg("home_team_score").alias("avg_points_for"),
                 F.avg("away_team_score").alias("avg_points_against"))
        )

    return df

# COMMAND ----------

@dlt.create_table(
    comment="Aggregated statistics by away team and season"
)
def team_stats_season_away_gold():
    # Read the data from the silver table
    df = spark.table("LIVE.Overall_Game_Silver")

    # Perform groupBy and aggregation operations
    df = (df.groupBy(["away_team_name", "season"])
            .agg(F.count("*").alias("count"),
                 F.avg("away_team_score").alias("avg_points_for"),
                 F.avg("home_team_score").alias("avg_points_against"))
         )

    return df