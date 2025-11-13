"""
Database repository for PDF processing results.
"""

import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, final

from ..core.types import ProcessingResult, PageData, TableData, TextLine, RestaurantRecord


@final
class PDFRepository:
  """Repository for storing and retrieving PDF processing results."""

  def __init__(self, db_path: str | Path) -> None:
    """
    Initialize repository with database path.

    Args:
        db_path: Path to SQLite database file
    """
    self.db_path = str(db_path)

  def _get_connection(self) -> sqlite3.Connection:
    """Get database connection."""
    conn = sqlite3.connect(self.db_path)
    conn.row_factory = sqlite3.Row
    return conn

  def save_processing_result(
    self,
    result: ProcessingResult,
    source_url: Optional[str] = None,
    download_date: Optional[datetime] = None,
  ) -> int:
    """
    Save complete processing result to database.

    Args:
        result: ProcessingResult to save
        source_url: Optional source URL for scraped PDFs
        download_date: Optional download timestamp

    Returns:
        Document ID
    """
    conn = self._get_connection()
    cursor = conn.cursor()

    try:
      # Insert document
      cursor.execute(
        """
        INSERT INTO documents (
          pdf_path, original_filename, source_url, download_date,
          total_pages, total_text_lines, total_tables,
          extraction_method, processing_time, status
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'completed')
        """,
        (
          result.pdf_path,
          Path(result.pdf_path).name,
          source_url,
          download_date.isoformat() if download_date else None,
          result.total_pages,
          result.total_text_lines,
          result.total_tables,
          result.extraction_method,
          result.processing_time,
        ),
      )
      document_id = cursor.lastrowid

      # Insert pages and their content
      for page in result.pages:
        page_id = self._save_page(cursor, document_id, page)
        self._save_text_lines(cursor, page_id, page.text_lines)
        self._save_tables(cursor, page_id, page.tables)

      conn.commit()
      return document_id

    except sqlite3.IntegrityError as e:
      conn.rollback()
      # Check if document already exists
      cursor.execute("SELECT id FROM documents WHERE pdf_path = ?", (result.pdf_path,))
      row = cursor.fetchone()
      if row:
        return row[0]
      raise RuntimeError(f"Failed to save processing result: {e}") from e
    finally:
      conn.close()

  def _save_page(self, cursor: sqlite3.Cursor, document_id: int, page: PageData) -> int:
    """Save page data."""
    cursor.execute(
      """
      INSERT INTO pages (document_id, page_number, total_lines, raw_text)
      VALUES (?, ?, ?, ?)
      """,
      (document_id, page.page, page.total_lines, page.raw_text),
    )
    return cursor.lastrowid

  def _save_text_lines(
    self, cursor: sqlite3.Cursor, page_id: int, text_lines: list[TextLine]
  ) -> None:
    """Save text lines."""
    for line in text_lines:
      bbox = line.bbox if line.bbox else (None, None, None, None)
      cursor.execute(
        """
        INSERT INTO text_lines (
          page_id, line_number, text, confidence,
          bbox_x1, bbox_y1, bbox_x2, bbox_y2,
          nearest_address, coordinate_x, coordinate_y
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
          page_id,
          line.line_number,
          line.text,
          line.confidence,
          bbox[0],
          bbox[1],
          bbox[2],
          bbox[3],
          line.nearest_address,
          line.x,
          line.y,
        ),
      )

  def _save_tables(
    self, cursor: sqlite3.Cursor, page_id: int, tables: list[TableData]
  ) -> None:
    """Save tables and their rows."""
    for table in tables:
      # Insert table metadata
      cursor.execute(
        """
        INSERT INTO tables (
          page_id, table_index, num_rows, num_columns,
          columns_json, address_column
        ) VALUES (?, ?, ?, ?, ?, ?)
        """,
        (
          page_id,
          table.table_index,
          table.shape[0],
          table.shape[1],
          json.dumps(table.columns),
          None,  # address_column can be added later if needed
        ),
      )
      table_id = cursor.lastrowid

      # Insert table rows
      for row_idx, row_data in enumerate(table.rows):
        # Extract common fields
        place_name = self._extract_place_name(row_data)
        approval_amount = self._extract_approval_amount(row_data)

        cursor.execute(
          """
          INSERT INTO table_rows (
            table_id, row_index, row_data_json,
            place_name, approval_amount,
            nearest_address, coordinate_x, coordinate_y
          ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
          """,
          (
            table_id,
            row_idx,
            json.dumps(row_data, ensure_ascii=False),
            place_name,
            approval_amount,
            row_data.get("nearest_address"),
            row_data.get("x"),
            row_data.get("y"),
          ),
        )

  def _extract_place_name(self, row_data: dict[str, Any]) -> Optional[str]:
    """Extract place name from row data."""
    # Common column names for place names
    place_columns = ["가맹점명", "가맹점", "상호", "매장", "장소", "place", "store", "shop"]
    for col in place_columns:
      if col in row_data:
        return str(row_data[col])
    return None

  def _extract_approval_amount(self, row_data: dict[str, Any]) -> Optional[int]:
    """Extract and convert approval amount from row data."""
    amount_columns = ["승인금액", "금액", "amount"]
    for col in amount_columns:
      if col in row_data:
        value = row_data[col]
        if isinstance(value, (int, float)):
          return int(value)
        if isinstance(value, str):
          # Remove commas and quotes
          cleaned = value.replace(",", "").replace("'", "").replace('"', "").strip()
          try:
            return int(cleaned)
          except ValueError:
            pass
    return None

  def document_exists(self, pdf_path: str) -> bool:
    """Check if document already exists in database."""
    conn = self._get_connection()
    cursor = conn.cursor()
    try:
      cursor.execute("SELECT id FROM documents WHERE pdf_path = ?", (pdf_path,))
      return cursor.fetchone() is not None
    finally:
      conn.close()

  def url_exists(self, url: str) -> bool:
    """Check if URL already exists in scraping queue."""
    conn = self._get_connection()
    cursor = conn.cursor()
    try:
      cursor.execute("SELECT id FROM scraping_queue WHERE url = ?", (url,))
      return cursor.fetchone() is not None
    finally:
      conn.close()

  def add_to_scraping_queue(self, url: str, title: Optional[str] = None) -> int:
    """
    Add URL to scraping queue.

    Args:
        url: URL to add
        title: Optional title

    Returns:
        Queue item ID
    """
    conn = self._get_connection()
    cursor = conn.cursor()
    try:
      cursor.execute(
        """
        INSERT INTO scraping_queue (url, title, status)
        VALUES (?, ?, 'pending')
        """,
        (url, title),
      )
      conn.commit()
      return cursor.lastrowid
    except sqlite3.IntegrityError:
      # URL already exists
      cursor.execute("SELECT id FROM scraping_queue WHERE url = ?", (url,))
      row = cursor.fetchone()
      return row[0] if row else -1
    finally:
      conn.close()

  def update_queue_status(
    self,
    url: str,
    status: str,
    download_path: Optional[str] = None,
    error_message: Optional[str] = None,
  ) -> None:
    """Update scraping queue status."""
    conn = self._get_connection()
    cursor = conn.cursor()
    try:
      cursor.execute(
        """
        UPDATE scraping_queue
        SET status = ?, download_path = ?, processed_date = ?,
            error_message = ?
        WHERE url = ?
        """,
        (status, download_path, datetime.now().isoformat(), error_message, url),
      )
      conn.commit()
    finally:
      conn.close()

  def get_pending_queue_items(self, limit: int = 100) -> list[dict[str, Any]]:
    """Get pending items from scraping queue."""
    conn = self._get_connection()
    cursor = conn.cursor()
    try:
      cursor.execute(
        """
        SELECT id, url, title
        FROM scraping_queue
        WHERE status = 'pending'
        ORDER BY added_date ASC
        LIMIT ?
        """,
        (limit,),
      )
      rows = cursor.fetchall()
      return [dict(row) for row in rows]
    finally:
      conn.close()

  def log_processing(
    self, document_id: Optional[int], log_level: str, message: str
  ) -> None:
    """Add processing log entry."""
    conn = self._get_connection()
    cursor = conn.cursor()
    try:
      cursor.execute(
        """
        INSERT INTO processing_logs (document_id, log_level, message)
        VALUES (?, ?, ?)
        """,
        (document_id, log_level, message),
      )
      conn.commit()
    finally:
      conn.close()

  def save_restaurants(self, entries: list[RestaurantRecord]) -> int:
    """Save restaurant records derived from LLM output."""
    if not entries:
      return 0

    conn = self._get_connection()
    cursor = conn.cursor()
    inserted = 0

    try:
      for entry in entries:
        cursor.execute(
          """
          INSERT INTO restaurants (name, address, coordinate_x, coordinate_y, source_pdf, source_url)
          VALUES (?, ?, ?, ?, ?, ?)
          ON CONFLICT(name, source_pdf, address) DO UPDATE SET
            coordinate_x = excluded.coordinate_x,
            coordinate_y = excluded.coordinate_y,
            source_url = COALESCE(excluded.source_url, source_url)
          """,
          (
            entry.name,
            entry.address,
            entry.x,
            entry.y,
            entry.source_pdf,
            entry.source_url,
          ),
        )
        inserted += 1

      conn.commit()
      return inserted
    finally:
      conn.close()
