# NBA Analytics Hackathon

## Introduction

Welcome to our NBA Analytics Hackathon project, organized by Compass Analytics and Databricks. Our aim was to utilize historical play-by-play data dating back to 1996 to predict NBA score margins for the 2022-2023 season. Participants collaborated as Data Engineers, Data Analysts, and Data Scientists, journeying through the entire data analytics pipeline.

## Data 

**Sources:**

- **Play-by-Play Data**: `Play_by_play_YYYY-YY.parquet` files provide play-by-play details for each respective season.
- **Sample Game Data**: `Example_game_data.xlsx` contains a sneak peek of the type of data available.
- **Data Field Descriptions**: `Data_Dictionary_Hackathon.xlsx` elucidates all the columns available in the datasets.

All of this data is securely stored in an S3 bucket.

## Approach

### 1. Data Engineering

Our data engineering team was responsible for:

- **Ingesting Data**: Transferred raw data from S3 to Databricks Delta tables.
- **Data Processing**: Employed the Delta Live Table pipeline to transform data across three layers (bronze, silver, gold) based on the Medallion Architecture.
- **Outputting Refined Data**: Delivered cleaned and aggregated datasets for downstream analytics.

**Output Tables:**
- `nba_games_bronze` - Pristine data, right off ingestion.
- `overall_game_silver` - Sanitized game-level information.
- `play_by_play_silver` - Neatened play-by-play records.
- `play_by_*_gold` tables (5 in total) - Granular event and athlete details.
- `team_stats_season_*_gold` tables (2 in total) - Cumulative team statistics.

### 2. Data Analysis

Our analysts:

- **Data Procurement**: Leveraged SQL within the Databricks lakehouse to compile datasets for examination.
- **Dashboard Construction**: Devised a graphical interface showcasing individual and team metrics along with trends over the season.
- **Data Interpretation**: Delved deep into the data to unearth crucial insights.

**Key Deliverables:**
- Interactive Lakehouse dashboard.
- Over 3 pivotal takeaways extracted from the datasets.

### 3. Data Science

Our data science aficionados:

- **Preliminary Analysis**: Embarked on exploratory data analysis (EDA) and feature engineering with the play-by-play datasets.
- **Model Development**: Utilized scikit-learn to devise regression models for score margin predictions. Enhanced model performance by employing grid search to pinpoint the optimal regression algorithm and its associated hyperparameters.
- **Performance Metrics**: Assessed model reliability using cross-validation techniques.
- **Score Predictions**: Forecasted outcomes on a test dataset and juxtaposed the predictions with real results.

**Essential Outputs:**
- Comprehensive notebooks covering the entirety of the data science process.
- A detailed comparison table contrasting predictions with actual results.

## Results 

Our best-performing model showcased a commendable Mean Absolute Error (MAE) of approximately 9 when predicting on the test set. 

The overall endeavor offered participants a hands-on experience with a genuine analytics workflow, maximizing the capabilities of Databricks and Delta Lake.

## Repository Contents

In this repository, you'll find:

- Original data files.
- Notebooks encapsulating the work of both our data engineering and data science teams.
- Downloads of our dashboards and the SQL queries used.

Thank you for exploring our project! We trust the insights and methodologies presented here will be of value.