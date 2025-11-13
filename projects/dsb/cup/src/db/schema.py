"""
Database schema definitions for PDF processing results.
"""

import sqlite3
from pathlib import Path
from typing import final


@final
class DatabaseSchema:
  """Database schema definitions."""

  CREATE_DOCUMENTS_TABLE = """
  CREATE TABLE IF NOT EXISTS documents (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    pdf_path TEXT NOT NULL,
    original_filename TEXT NOT NULL,
    source_url TEXT,
    download_date TIMESTAMP,
    total_pages INTEGER,
    total_text_lines INTEGER,
    total_tables INTEGER,
    extraction_method TEXT,
    processing_time REAL,
    processing_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    status TEXT DEFAULT 'pending',
    error_message TEXT,
    UNIQUE(pdf_path)
  );
  """

  CREATE_PAGES_TABLE = """
  CREATE TABLE IF NOT EXISTS pages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    document_id INTEGER NOT NULL,
    page_number INTEGER NOT NULL,
    total_lines INTEGER,
    raw_text TEXT,
    FOREIGN KEY (document_id) REFERENCES documents(id) ON DELETE CASCADE,
    UNIQUE(document_id, page_number)
  );
  """

  CREATE_TEXT_LINES_TABLE = """
  CREATE TABLE IF NOT EXISTS text_lines (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    page_id INTEGER NOT NULL,
    line_number INTEGER NOT NULL,
    text TEXT NOT NULL,
    confidence REAL,
    bbox_x1 REAL,
    bbox_y1 REAL,
    bbox_x2 REAL,
    bbox_y2 REAL,
    nearest_address TEXT,
    coordinate_x REAL,
    coordinate_y REAL,
    FOREIGN KEY (page_id) REFERENCES pages(id) ON DELETE CASCADE
  );
  """

  CREATE_TABLES_TABLE = """
  CREATE TABLE IF NOT EXISTS tables (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    page_id INTEGER NOT NULL,
    table_index INTEGER NOT NULL,
    num_rows INTEGER,
    num_columns INTEGER,
    columns_json TEXT,
    address_column TEXT,
    FOREIGN KEY (page_id) REFERENCES pages(id) ON DELETE CASCADE,
    UNIQUE(page_id, table_index)
  );
  """

  CREATE_TABLE_ROWS_TABLE = """
  CREATE TABLE IF NOT EXISTS table_rows (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    table_id INTEGER NOT NULL,
    row_index INTEGER NOT NULL,
    row_data_json TEXT,
    place_name TEXT,
    approval_amount INTEGER,
    nearest_address TEXT,
    coordinate_x REAL,
    coordinate_y REAL,
    FOREIGN KEY (table_id) REFERENCES tables(id) ON DELETE CASCADE
  );
  """

  CREATE_RESTAURANTS_TABLE = """
  CREATE TABLE IF NOT EXISTS restaurants (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    address TEXT,
    coordinate_x REAL,
    coordinate_y REAL,
    source_pdf TEXT,
    source_url TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(name, source_pdf, address)
  );
  """

  CREATE_SCRAPING_QUEUE_TABLE = """
  CREATE TABLE IF NOT EXISTS scraping_queue (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    url TEXT NOT NULL UNIQUE,
    title TEXT,
    status TEXT DEFAULT 'pending',
    download_path TEXT,
    added_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    processed_date TIMESTAMP,
    error_message TEXT
  );
  """

  CREATE_PROCESSING_LOGS_TABLE = """
  CREATE TABLE IF NOT EXISTS processing_logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    document_id INTEGER,
    log_level TEXT,
    message TEXT,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (document_id) REFERENCES documents(id) ON DELETE CASCADE
  );
  """

  # Indexes for performance
  CREATE_INDEXES = [
    "CREATE INDEX IF NOT EXISTS idx_documents_status ON documents(status);",
    "CREATE INDEX IF NOT EXISTS idx_documents_source_url ON documents(source_url);",
    "CREATE INDEX IF NOT EXISTS idx_pages_document_id ON pages(document_id);",
    "CREATE INDEX IF NOT EXISTS idx_text_lines_page_id ON text_lines(page_id);",
    "CREATE INDEX IF NOT EXISTS idx_tables_page_id ON tables(page_id);",
    "CREATE INDEX IF NOT EXISTS idx_table_rows_table_id ON table_rows(table_id);",
    "CREATE INDEX IF NOT EXISTS idx_table_rows_place_name ON table_rows(place_name);",
    "CREATE INDEX IF NOT EXISTS idx_scraping_queue_status ON scraping_queue(status);",
    "CREATE INDEX IF NOT EXISTS idx_processing_logs_document_id ON processing_logs(document_id);",
    "CREATE INDEX IF NOT EXISTS idx_restaurants_name ON restaurants(name);",
  ]


def init_database(db_path: str | Path) -> None:
  """
  Initialize database with schema.

  Args:
      db_path: Path to SQLite database file
  """
  db_path = Path(db_path)
  db_path.parent.mkdir(parents=True, exist_ok=True)

  conn = sqlite3.connect(str(db_path))
  cursor = conn.cursor()

  try:
    # Create tables
    cursor.execute(DatabaseSchema.CREATE_DOCUMENTS_TABLE)
    cursor.execute(DatabaseSchema.CREATE_PAGES_TABLE)
    cursor.execute(DatabaseSchema.CREATE_TEXT_LINES_TABLE)
    cursor.execute(DatabaseSchema.CREATE_TABLES_TABLE)
    cursor.execute(DatabaseSchema.CREATE_TABLE_ROWS_TABLE)
    cursor.execute(DatabaseSchema.CREATE_SCRAPING_QUEUE_TABLE)
    cursor.execute(DatabaseSchema.CREATE_PROCESSING_LOGS_TABLE)
    cursor.execute(DatabaseSchema.CREATE_RESTAURANTS_TABLE)

    # Create indexes
    for index_sql in DatabaseSchema.CREATE_INDEXES:
      cursor.execute(index_sql)

    conn.commit()
  finally:
    conn.close()
