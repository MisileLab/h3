import os
import sys
import asyncio
from pathlib import Path

from dotenv import load_dotenv

# Add the parent directory to Python path
sys.path.append(str(Path(__file__).parent.parent))

from src.core.kakao_api import search_places, convert_coordinates, calculate_distance

async def search_and_display(
  api_key: str,
  query: str,
  page: int = 1,
  ref_wtm_x: float | None = None,
  ref_wtm_y: float | None = None,
  sort_by_distance: bool = False,
  nearest_only: bool = False
) -> bool:
  """
  Search for places and display results.
  
  Args:
      api_key: Kakao API key
      query: Search query
      page: Page number
      ref_wtm_x: Reference X coordinate in WTM format
      ref_wtm_y: Reference Y coordinate in WTM format
      sort_by_distance: Whether to sort results by distance
      nearest_only: Whether to show only the nearest place
  
  Returns:
      Whether there are more pages
  """
  try:
    results = await search_places(
      api_key, 
      query, 
      page=page, 
      ref_wtm_x=ref_wtm_x, 
      ref_wtm_y=ref_wtm_y, 
      nearest_only=nearest_only
    )
    
    # Calculate distances if reference coordinates are provided and not already calculated
    if ref_wtm_x is not None and ref_wtm_y is not None and not nearest_only:
      print(f"\nReference coordinates (WTM): {ref_wtm_x}, {ref_wtm_y}")
      
      # Calculate distance for each place
      for place in results.documents:
        place_wtm_x, place_wtm_y = await convert_coordinates(api_key, str(place.x), str(place.y))
        place.calculated_distance = calculate_distance(ref_wtm_x, ref_wtm_y, place_wtm_x, place_wtm_y)
      
      # Sort by distance if requested
      if sort_by_distance:
        results.documents.sort(key=lambda p: p.calculated_distance or float('inf'))
    
    print(f"\nFound {results.meta.total_count} total places")
    print(f"Showing {len(results.documents)} results from page {page}\n")
    
    for i, place in enumerate(results.documents, 1):
      print(f"{i}. 🏢 {place.place_name}")
      print(f"📍 {place.road_address_name}")
      print(f"📞 {place.phone}")
      print(f"🔗 {place.place_url}")
      print(f"📍 Coordinates: {place.x}, {place.y}")
      
      if place.calculated_distance is not None:
        print(f"📏 Distance: {place.calculated_distance:.2f} meters")
        
      print("-" * 50)
    
    print(f"\nMore pages available: {'No' if results.meta.is_end else 'Yes'}")
    return not results.meta.is_end
    
  except Exception as e:
    print(f"Error occurred: {e}")
    return False

async def main():
  # Load environment variables
  _ = load_dotenv()
  
  api_key = os.getenv("KAKAO_API_KEY")
  if not api_key:
    print("Error: KAKAO_API_KEY environment variable not set")
    return

  # Ask user if they want to use location-based search
  use_location = input("\nDo you want to search based on distance from a location? (y/n): ").strip().lower() == 'y'
  
  ref_wtm_x = None
  ref_wtm_y = None
  
  if use_location:
    print("\nEnter reference coordinates:")
    ref_wtm_x = float(input("wtm_x: ").strip())
    ref_wtm_y = float(input("wtm_y: ").strip())
    print(f"Using reference coordinates: {ref_wtm_x}, {ref_wtm_y}")

  while True:
    # Get search query from user
    query = input("\nEnter search query (or 'q' to quit): ").strip()
    if query.lower() == 'q':
      break
    
    # Ask if results should be sorted by distance or show only nearest
    sort_by_distance = False
    nearest_only = False
    
    if use_location:
      location_option = input("\nOptions:\n1. Sort by distance\n2. Show only nearest place\n3. Show all unsorted\nChoose (1/2/3): ").strip()
      
      if location_option == '1':
        sort_by_distance = True
      elif location_option == '2':
        nearest_only = True
    
    page = 1
    has_more = True
    
    while has_more:
      has_more = await search_and_display(
        api_key, 
        query, 
        page, 
        ref_wtm_x=ref_wtm_x, 
        ref_wtm_y=ref_wtm_y, 
        sort_by_distance=sort_by_distance,
        nearest_only=nearest_only
      )
      
      if has_more:
        next_page = input("\nPress Enter for next page or 'n' for new search: ").strip()
        if next_page.lower() == 'n':
          break
        page += 1
      else:
        _ = input("\nPress Enter to continue...")

  print("\nThank you for using the search service!")

if __name__ == "__main__":
  asyncio.run(main())