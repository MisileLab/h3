"""
JSON output formatter.
"""

import os
import json
import asyncio
from typing import override, final

from rich.console import Console

from .base import BaseOutputFormatter
from ..core.types import ProcessingResult
from ..core.exceptions import OutputError
from ..core.kakao_api import search_places

console = Console()


@final
class JSONFormatter(BaseOutputFormatter):
  """Format processing results as JSON."""
  
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
          # Look specifically for "가맹점명" column which contains place names
          place_column = "가맹점명" if "가맹점명" in table.columns else None
          
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
            # Process each row in the table
            for row in table.rows:
              place_name = row.get(place_column)
              if place_name and isinstance(place_name, str) and place_name.strip():
                # Skip if we already have this place
                if place_name in address_info:
                  continue
                  
                try:
                  # Search for the place using Kakao API
                  search_result = await search_places(
                    api_key,
                    place_name,
                    nearest_only=True,
                    ref_wtm_x=self.ref_wtm_x,
                    ref_wtm_y=self.ref_wtm_y
                  )
                  
                  # If we found a place, store its information
                  if search_result.documents:
                    place = search_result.documents[0]
                    address_info[place_name] = (place.address_name, place.x, place.y)
                except Exception as e:
                  console.print(f"⚠️ Error getting address info for place: {place_name} - {str(e)}", style="yellow")
    
    except Exception as e:
      console.print(f"⚠️ Error processing tables for address information: {str(e)}", style="yellow")
    
    return address_info
  
  async def _add_address_info(self, result: ProcessingResult) -> ProcessingResult:
    """Add address and coordinate information to text lines."""
    api_key = os.getenv("KAKAO_API_KEY")
    if not api_key:
      console.print("⚠️ KAKAO_API_KEY not found in environment. Address information will not be added.", style="yellow")
      return result
    
    try:
      # First, get address information from tables
      address_info = await self._add_address_info_from_tables(result)
      
      # Apply address information to text lines
      for page in result.pages:
        for line in page.text_lines:
          # Check if the text contains any place names from the tables
          place_found = False
          for place_name, (address, x, y) in address_info.items():
            if place_name in line.text:
              line.nearest_address = address
              line.x = x
              line.y = y
              place_found = True
              break
          
          # If no place found in the tables, use the text content as search query
          if not place_found and line.text and len(line.text.strip()) > 0:
            search_query = line.text.strip()
            try:
              # Search for places using the text content
              search_result = await search_places(
                api_key, 
                search_query, 
                nearest_only=True,
                ref_wtm_x=self.ref_wtm_x,
                ref_wtm_y=self.ref_wtm_y
              )
              
              # If we found a place, add its information to the line
              if search_result.documents:
                place = search_result.documents[0]
                # Add address and coordinates as additional attributes
                line.nearest_address = place.address_name
                line.x = place.x
                line.y = place.y
            except Exception as e:
              console.print(f"⚠️ Error getting address info for text: {search_query[:30]}... - {str(e)}", style="yellow")
    except Exception as e:
      console.print(f"⚠️ Error adding address information: {str(e)}", style="yellow")
    
    return result

  @override
  def format(self, result: ProcessingResult) -> str:
    """Format the processing result as JSON."""
    # Add address information if possible
    try:
      result = asyncio.run(self._add_address_info(result))
    except Exception as e:
      console.print(f"⚠️ Could not add address information: {str(e)}", style="yellow")
    
    # Convert Pydantic models to dictionaries using model_dump()
    result_dict = result.model_dump()
    
    return json.dumps(result_dict, ensure_ascii=False, indent=2)
  
  @override
  def save(self, result: ProcessingResult, output_path: str) -> None:
    """Save the processing result as JSON."""
    try:
      self.ensure_directory(output_path)
      
      json_content = self.format(result)
      with open(output_path, "w", encoding="utf-8") as f:
        f.write(json_content)
      
      console.print(f"✅ JSON saved to: {output_path}", style="green")
      
    except (OSError, IOError, ValueError) as e:
      raise OutputError(f"Error saving JSON: {e}") from e 