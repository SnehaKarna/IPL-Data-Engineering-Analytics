# Project data

The app uses small, versioned Gold snapshots in `data/gold/` for local development. The Streamlit Overview uses the CSV exports in this folder. These files are intentionally kept in Git so a clone can run without Databricks access.

The original raw CSV datasets and full Silver history are not stored in this repository. Databricks source and Delta data use the Unity Catalog Volume:

`/Volumes/workspace/default/ipl_data`

The retained pipeline expects raw CSVs in `Raw/` and writes Silver plus its current Gold outputs under the same Volume. Gold exports are not synchronized automatically: copy refreshed Parquet snapshots into `data/gold/` when available, then rerun `python -m rag.vector_store` to refresh Qdrant.

See the root README for local app setup and the Spark pipeline's eight Gold outputs. The checked-in local snapshots currently cover the six tables consumed by the assistant.
