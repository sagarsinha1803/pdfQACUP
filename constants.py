import os
from chromadb.config import Settings

# Chrome Settings
CHROMA_DB_SETTINGS = Settings(
    chroma_db_impl = 'duckdb+parquet',
    persist_directory = 'db',
    anonymized_telemetry = False
)