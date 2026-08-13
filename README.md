# IPL Data Engineering & Analytics Pipeline

An end-to-end IPL data engineering and analytics project built using **AWS S3, Databricks, Apache Spark, PySpark, Spark SQL, and Streamlit**.

The project processes IPL ball-by-ball and supporting datasets through a **Raw → Silver → Gold Medallion Architecture**, performs analytical transformations using PySpark and Spark SQL, and presents key insights through an interactive dashboard.

## Architecture

```text
IPL CSV Datasets
       │
       ▼
   AWS S3
   Raw Layer
       │
       ▼
  Databricks
 Apache Spark
    PySpark
       │
       ▼
  Silver Layer
 Optimized Parquet
       │
       ▼
   Gold Layer
 ┌───────────────┐
 │ Top Batsmen   │
 │ Top Bowlers   │
 │ Team Results  │
 │ Toss Analysis │
 └───────────────┘
       │
       ▼
   Spark SQL
       │
       ▼
 Streamlit Dashboard
```

## Dataset

The project analyzes IPL data containing:

* **295K+ ball-by-ball deliveries**
* **1,243 matches**
* **261 players**
* **19 teams**

The raw datasets include match information, ball-by-ball delivery data, player details, and team information.

Raw data is stored in Amazon S3 and is not included directly in this repository.

## Technology Stack

* **Cloud Storage:** AWS S3
* **Data Engineering:** Apache Spark, PySpark
* **Data Platform:** Databricks
* **Analytics:** Spark SQL
* **Visualization:** Streamlit, Plotly
* **Programming:** Python, SQL
* **Version Control:** Git, GitHub

## Data Pipeline

### 1. Raw Layer

IPL CSV datasets are uploaded to Amazon S3:

```text
s3://ipl-2010-data/Raw/
```

Datasets include:

* Player details
* Match information
* Ball-by-ball match data
* Team information

### 2. Silver Layer

The raw datasets are loaded into Databricks using PySpark.

Data cleaning and transformation includes:

* Schema validation
* Data type handling
* Null-value handling
* Dataset transformations
* Joining related datasets
* Validation of match and delivery IDs

The processed data is stored as optimized Parquet data.

### 3. Gold Layer

Analytical datasets are created for:

* Top batsmen
* Top bowlers
* Team performance
* Toss analysis

These datasets are queried using Spark SQL.

## Key Analytics

### Top Batsmen

Identifies the highest run scorers across the analyzed IPL matches.

### Top Bowlers

Ranks bowlers based on total wickets.

### Team Performance

Analyzes team-level match performance and win percentages.

### Toss Analysis

Analyzes whether choosing to bat or field after winning the toss is associated with higher match-winning percentages.

Example result:

| Toss Decision | Matches | Toss Winner Wins |  Win % |
| ------------- | ------: | ---------------: | -----: |
| Field         |     825 |              443 | 53.70% |
| Bat           |     418 |              185 | 44.26% |

## Dashboard

The Streamlit dashboard provides an interactive presentation of the Gold-layer analytics, including:

* IPL dataset KPIs
* Top 10 run scorers
* Top 10 wicket takers
* Team win percentage
* Toss decision analysis

## Repository Structure

```text
IPL_ANALYSIS/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── notebooks/
│   └── IPL_DATA_ANALYSIS_SPARK.py
│
├── dashboard/
│   └── app.py
│
├── data/
│   └── README.md
│
└── images/
    └── README.md
```

## How to Run the Dashboard

Install the required Python packages:

```bash
pip install -r requirements.txt
```

Run Streamlit:

```bash
python -m streamlit run dashboard/app.py
```

The dashboard will open locally in your browser.

## Key Outcomes

* Built an end-to-end cloud-based data pipeline using AWS S3 and Databricks.
* Processed 295K+ IPL ball-by-ball records using PySpark.
* Implemented Raw → Silver → Gold Medallion Architecture.
* Used Parquet for processed analytical datasets.
* Developed Spark SQL queries for player, team, and match-level analytics.
* Built an interactive Streamlit dashboard for presenting analytical results.

## Future Improvements

* Add automated ETL orchestration using Apache Airflow.
* Add incremental data ingestion.
* Implement data-quality validation checks.
* Deploy the Streamlit dashboard publicly.
* Add additional IPL season and player-level analytics.

## Author

**Sneha Karna**

Computer Science Engineering
Data Engineering | Cloud | Analytics
