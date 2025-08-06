"""
CSV output formatter.
"""

import os
import polars as pl
import asyncio
from pathlib import Path
from typing import override, final, Any

from rich.console import Console

from .base import BaseOutputFormatter
from ..core.types import ProcessingResult, CSVRow
from ..core.exceptions import OutputError
from ..core.kakao_api import search_places

console = Console()

# Mapping dictionary for place names that need translation or alternative spellings
place_name_mapping = {
  # English to Korean
  "wagamama": "와가마마"
}

@final
class CSVFormatter(BaseOutputFormatter):
  """Format processing results as CSV."""
  
  async def _add_address_info_from_tables(self, result: ProcessingResult) -> dict[str, tuple[str, float, float]]:
    """
    Extract address information from tables in the result.
    
    Args:
        result: The processing result containing tables
        
    Returns:
        Dictionary mapping place names to (address, x, y) tuples
    """
    api_key = os.getenv("KAKAO_API_KEY")
    if not api_key:
      console.print("⚠️ KAKAO_API_KEY not found in environment. Address information will not be added.", style="yellow")
      return {}
    
    address_info = {}
    
    try:
      # Look for tables in the result
      for page in result.pages:
        for table in page.tables:
          # Print table columns for debugging
          console.print(f"\n📋 Table columns: {table.columns}", style="blue")
          
          # Print first few rows for debugging
          console.print(f"\n📋 Sample table rows:", style="blue")
          for i, row in enumerate(table.rows[:3]):
            console.print(f"  • Row {i+1}: {row}", style="blue")
          
          # Look specifically for "가맹점명" column which contains place names
          place_column = "가맹점명" if "가맹점명" in table.columns else None
          
          if place_column:
            console.print(f"✅ Found '가맹점명' column in table", style="green")
            # Show sample values from this column
            sample_values = [row.get(place_column, "") for row in table.rows[:5]]
            console.print(f"📋 Sample values from '가맹점명' column:", style="blue")
            for i, val in enumerate(sample_values):
              console.print(f"  • Value {i+1}: '{val}'", style="blue")
          else:
            console.print(f"⚠️ '가맹점명' column not found in table", style="yellow")
          
          # If "가맹점명" not found, try other likely column names
          if not place_column:
            for col in table.columns:
              if any(keyword in col.lower() for keyword in ["가맹점", "상호", "매장", "장소", "place", "store", "shop"]):
                place_column = col
                break
            
            if not place_column and len(table.columns) > 0:
              # If no obvious place column found, use the first one that might contain place names
              for col in table.columns:
                sample_values = [str(row.get(col, "")) for row in table.rows[:5] if row.get(col)]
                if any(len(val) > 0 and len(val) < 30 for val in sample_values):  # Likely place names
                  place_column = col
                  break
          
          if place_column:
            console.print(f"🔍 Found place column: '{place_column}'", style="blue")
            # Process each row in the table
            for row in table.rows:
              place_name = row.get(place_column)
              if place_name and isinstance(place_name, str) and place_name.strip():
                console.print(f"🏢 Processing place: '{place_name}'", style="blue")
                # Skip if we already have this place
                if place_name in address_info:
                  console.print(f"⏭️ Skipping duplicate place: '{place_name}'", style="blue")
                  continue
                  
                try:
                  # Search for the place using Kakao API
                  console.print(f"🔍 Searching for place: '{place_name}' with ref coordinates: ({self.ref_wtm_x}, {self.ref_wtm_y})", style="blue")
                  
                  search_query = place_name_mapping.get(place_name, place_name)
                  if search_query != place_name:
                    console.print(f"🔄 Mapped '{place_name}' to '{search_query}' for search", style="blue")
                  
                  search_result = await search_places(
                    api_key,
                    search_query,
                    nearest_only=True,
                    ref_wtm_x=self.ref_wtm_x,
                    ref_wtm_y=self.ref_wtm_y
                  )
                  
                  # If we found a place, store its information
                  if search_result.documents:
                    place = search_result.documents[0]
                    address_info[place_name] = (place.address_name, place.x, place.y)
                    console.print(f"✅ Found address for '{place_name}': {place.address_name} ({place.x}, {place.y})", style="green")
                  else:
                    console.print(f"❌ No results found for place: '{place_name}'", style="red")
                except Exception as e:
                  console.print(f"⚠️ Error getting address info for place: {place_name} - {str(e)}", style="yellow")
    
    except Exception as e:
      console.print(f"⚠️ Error processing tables for address information: {str(e)}", style="yellow")
    
    return address_info
  
  async def _add_address_info(self, csv_rows: list[CSVRow], result: ProcessingResult) -> list[CSVRow]:
    """Add address and coordinate information to CSV rows."""
    console.print("\n📍 Starting address lookup process...", style="blue")
    
    api_key = os.getenv("KAKAO_API_KEY")
    if not api_key:
      console.print("⚠️ KAKAO_API_KEY not found in environment. Address information will not be added.", style="yellow")
      return csv_rows
    else:
      console.print("✅ KAKAO_API_KEY found in environment", style="green")
    
    try:
      # First, get address information from tables
      console.print("🔍 Extracting address information from tables...", style="blue")
      address_info = await self._add_address_info_from_tables(result)
      console.print(f"📊 Found {len(address_info)} unique places in tables", style="blue")
      
      # Apply address information to CSV rows
      console.print("🔄 Applying address information to CSV rows...", style="blue")
      for row in csv_rows:
        # Check if the text contains any place names from the tables
        place_found = False
        for place_name, (address, x, y) in address_info.items():
          if place_name in row.text:
            row.nearest_address = address
            row.x = x
            row.y = y
            place_found = True
            console.print(f"✅ Applied address info for '{place_name}' to row", style="green")
            break
        
        # If no place found in the tables, use the text content as search query
        if not place_found and row.text and len(row.text.strip()) > 0:
          original_text = row.text.strip()
          
          # Try to find place names in the text that match our mapping
          search_query = original_text
          for place_name, mapped_name in place_name_mapping.items():
            if place_name in original_text:
              # Replace the place name in the search query
              search_query = original_text.replace(place_name, mapped_name)
              console.print(f"🔄 Found '{place_name}' in text, using '{mapped_name}' for search", style="blue")
              break
              
          try:
            # Search for places using the text content
            console.print(f"🔍 Searching using text: '{search_query}'", style="blue")
            search_result = await search_places(
              api_key, 
              search_query, 
              nearest_only=True,
              ref_wtm_x=self.ref_wtm_x,
              ref_wtm_y=self.ref_wtm_y
            )
            
            # If we found a place, add its information to the row
            if search_result.documents:
              place = search_result.documents[0]
              row.nearest_address = place.address_name
              row.x = place.x
              row.y = place.y
          except Exception as e:
            console.print(f"⚠️ Error getting address info for text: {search_query[:30]}... - {str(e)}", style="yellow")
    except Exception as e:
      console.print(f"⚠️ Error adding address information: {str(e)}", style="yellow")
    
    return csv_rows

  @override
  def format(self, result: ProcessingResult) -> str:
    """Format the processing result as CSV."""
    csv_rows = []

    for page in result.pages:
      page_num = page.page
      for line in page.text_lines:
        csv_row = CSVRow(
          page=page_num,
          line=line.line_number,
          text=line.text,
          confidence=line.confidence or 0.0,
          nearest_address=None,
          x=None,
          y=None
        )
        csv_rows.append(csv_row)
    
    # Add address information if possible
    try:
      csv_rows = asyncio.run(self._add_address_info(csv_rows, result))
    except Exception as e:
      console.print(f"⚠️ Could not add address information: {str(e)}", style="yellow")
    
    # Check if any address information was found
    address_count = sum(1 for row in csv_rows if row.nearest_address is not None)
    console.print(f"\n📊 Address information summary:", style="blue")
    console.print(f"  • Total rows: {len(csv_rows)}", style="blue")
    console.print(f"  • Rows with address info: {address_count}", style="blue")
    
    if address_count > 0:
        console.print(f"\n✅ Successfully added address information to {address_count} rows", style="green")
        # Show a sample of rows with address info
        for i, row in enumerate([r for r in csv_rows if r.nearest_address is not None][:3]):
            console.print(f"  • Row {i+1}: {row.text[:30]}... -> {row.nearest_address} ({row.x}, {row.y})", style="green")
    else:
        console.print(f"\n⚠️ No address information was found for any rows", style="yellow")
        console.print(f"  • Check that KAKAO_API_KEY is set correctly", style="yellow")
        console.print(f"  • Check that place names in the '가맹점명' column are valid", style="yellow")
        console.print(f"  • Check that reference coordinates are valid", style="yellow")
    
    # Convert Pydantic models to dictionaries for Polars
    rows_data = [row.model_dump() for row in csv_rows]
    df = pl.DataFrame(rows_data)
    return df.write_csv()
  
  @override
  def save(self, result: ProcessingResult, output_path: str) -> None:
    """Save the processing result as CSV."""
    try:
      self.ensure_directory(output_path)
      
      # Save text data
      csv_content = self.format(result)
      with open(output_path, "w", encoding="utf-8") as f:
        f.write(csv_content)
      
      console.print(f"✅ CSV saved to: {output_path}", style="green")
      
      # Save tables if they exist
      self._save_tables(result, output_path)
      
    except (OSError, IOError, ValueError) as e:
      raise OutputError(f"Error saving CSV: {e}") from e
  
  def _convert_amount_to_int(self, table_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Convert the '승인금액' column from string to integer.
    
    Args:
        table_rows: List of table rows as dictionaries
        
    Returns:
        Updated table rows with integer amounts
    """
    updated_rows = []
    for row in table_rows:
      updated_row = row.copy()
      
      # Check if the row has a '승인금액' column
      if "승인금액" in updated_row:
        amount_str = updated_row["승인금액"]
        if isinstance(amount_str, str):
          # Remove commas and convert to integer
          try:
            # Remove quotes, commas and other non-numeric characters
            clean_amount = amount_str.replace('"', '').replace(',', '').replace(' ', '')
            updated_row["승인금액"] = int(clean_amount)
            console.print(f"✅ Converted amount '{amount_str}' to {updated_row['승인금액']}", style="green")
          except ValueError:
            console.print(f"⚠️ Could not convert amount '{amount_str}' to integer", style="yellow")
      
      updated_rows.append(updated_row)
    
    return updated_rows

  async def _add_address_info_to_table_rows(self, table_rows: list[dict[str, Any]], place_column: str | None = None) -> list[dict[str, Any]]:
    """
    Add address and coordinate information to table rows.
    
    Args:
        table_rows: List of table rows as dictionaries
        place_column: Column name containing place names (e.g., "가맹점명")
        
    Returns:
        Updated table rows with address information
    """
    api_key = os.getenv("KAKAO_API_KEY")
    if not api_key:
      console.print("⚠️ KAKAO_API_KEY not found in environment. Address information will not be added to tables.", style="yellow")
      return table_rows
    
    # If no place column specified, try to find it
    if not place_column:
      # Get all column names from the first row (if available)
      if table_rows and len(table_rows) > 0:
        columns = list(table_rows[0].keys())
        
        # Look for "가맹점명" column
        if "가맹점명" in columns:
          place_column = "가맹점명"
        else:
          # Try to find a suitable column
          for col in columns:
            if any(keyword in col.lower() for keyword in ["가맹점", "상호", "매장", "장소", "place", "store", "shop"]):
              place_column = col
              break
    
    if not place_column:
      console.print("⚠️ Could not find place name column in table.", style="yellow")
      return table_rows
    
    console.print(f"🔍 Using '{place_column}' column for place names in table", style="blue")
    
    # Process each row
    updated_rows = []
    for row in table_rows:
      # Create a copy of the row to avoid modifying the original
      updated_row = row.copy()
      
      # Get place name from the row
      place_name = row.get(place_column)
      if place_name and isinstance(place_name, str) and place_name.strip():
        try:
          # Get search query from mapping or use original place name
          original_place_name = place_name.strip()
          console.print(f"original_place_name: {original_place_name}")
          search_query = place_name_mapping.get(original_place_name, original_place_name)
          
          if search_query != original_place_name:
            console.print(f"🔄 Mapped '{original_place_name}' to '{search_query}' for search", style="blue")
          
          # Search for the place using Kakao API
          console.print(f"🔍 Searching for place in table: '{search_query}'", style="blue")
          search_result = await search_places(
            api_key,
            search_query,
            nearest_only=True,
            ref_wtm_x=self.ref_wtm_x,
            ref_wtm_y=self.ref_wtm_y
          )
          
          # If we found a place, add its information to the row
          if search_result.documents:
            place = search_result.documents[0]
            updated_row["nearest_address"] = place.address_name
            updated_row["x"] = float(place.x)
            updated_row["y"] = float(place.y)
            console.print(f"✅ Found address for table row '{place_name}': {place.address_name}", style="green")
          else:
            console.print(f"⚠️ No address found for table row '{place_name}'", style="yellow")
        except Exception as e:
          console.print(f"⚠️ Error getting address info for table row '{place_name}': {str(e)}", style="yellow")
      
      updated_rows.append(updated_row)
    
    return updated_rows

  def _save_tables(self, result: ProcessingResult, base_path: str) -> None:
    """Save tables to separate CSV files."""
    all_tables = []
    for page in result.pages:
      all_tables.extend(page.tables)
    
    if not all_tables:
      return
    
    base_path_obj = Path(base_path)
    base_name = base_path_obj.stem
    base_dir = base_path_obj.parent
    
    # For single table, save in normal table format
    if len(all_tables) == 1:
      table = all_tables[0]
      # Add address information to table rows
      updated_rows = asyncio.run(self._add_address_info_to_table_rows(table.rows, "가맹점명"))
      # Convert 승인금액 to integer
      updated_rows = self._convert_amount_to_int(updated_rows)
      df = pl.DataFrame(updated_rows)
      table_path = base_dir / f"{base_name}_table.csv"
      df.write_csv(table_path)
      console.print(f"✅ Table CSV saved to: {table_path}", style="green")
    else:
      # For multiple tables, save each to separate files
      for table in all_tables:
        table_idx = table.table_index
        # Add address information to table rows
        updated_rows = asyncio.run(self._add_address_info_to_table_rows(table.rows))
        # Convert 승인금액 to integer
        updated_rows = self._convert_amount_to_int(updated_rows)
        df = pl.DataFrame(updated_rows)
        table_path = base_dir / f"{base_name}_table_{table_idx}.csv"
        df.write_csv(table_path)
        console.print(f"✅ Table {table_idx} CSV saved to: {table_path}", style="green")
      
      # Also create a combined file with table identification
      combined_rows = []
      for table in all_tables:
        table_idx = table.table_index
        # Add address information to table rows
        updated_rows = asyncio.run(self._add_address_info_to_table_rows(table.rows))
        # Convert 승인금액 to integer
        updated_rows = self._convert_amount_to_int(updated_rows)
        for row in updated_rows:
          row_with_table = {"Table": f"Table_{table_idx}", **row}
          combined_rows.append(row_with_table)
      
      combined_df = pl.DataFrame(combined_rows)
      combined_path = base_dir / f"{base_name}_combined.csv"
      combined_df.write_csv(combined_path)
      console.print(f"✅ Combined tables CSV saved to: {combined_path}", style="green") 