import urllib.parse
import math
from pydantic import BaseModel, HttpUrl
import httpx

class KakaoPlace(BaseModel):
  address_name: str
  category_group_code: str
  category_group_name: str
  category_name: str
  distance: str
  id: str
  phone: str
  place_name: str
  place_url: HttpUrl
  road_address_name: str
  x: float  # longitude
  y: float  # latitude
  calculated_distance: float | None = None  # Added field for distance calculation

class KakaoSearchMeta(BaseModel):
  is_end: bool
  pageable_count: int
  same_name: dict[str, str | list[str] | dict[str, str]]
  total_count: int

class KakaoSearchResponse(BaseModel):
  documents: list[KakaoPlace]
  meta: KakaoSearchMeta

async def convert_coordinates(
  api_key: str,
  x: str,
  y: str,
  output_coord: str = "WTM"
) -> tuple[float, float]:
  """
  Convert coordinates using Kakao Local API.

  Args:
      api_key: Kakao REST API key
      x: X coordinate (longitude)
      y: Y coordinate (latitude)
      output_coord: Output coordinate system (default: WTM)

  Returns:
      Tuple of (x, y) coordinates in the target system

  Raises:
      httpx.HTTPError: If the API request fails
  """
  from rich.console import Console
  console = Console()
  
  url = "https://dapi.kakao.com/v2/local/geo/transcoord.json"
  params = {"x": x, "y": y, "output_coord": output_coord}
  headers = {"Authorization": f"KakaoAK {api_key}"}

  console.print(f"🔄 Converting coordinates ({x}, {y}) to {output_coord} format", style="blue")
  
  async with httpx.AsyncClient() as client:
    response = await client.get(url, params=params, headers=headers)
    response.raise_for_status()
    data = response.json()
    x_coord = float(data["documents"][0]["x"])
    y_coord = float(data["documents"][0]["y"])
    console.print(f"✅ Converted coordinates: ({x_coord}, {y_coord})", style="green")
    return x_coord, y_coord

def calculate_distance(x1: float, y1: float, x2: float, y2: float) -> float:
  """
  Calculate Euclidean distance between two points.

  Args:
      x1: X coordinate of first point
      y1: Y coordinate of first point
      x2: X coordinate of second point
      y2: Y coordinate of second point

  Returns:
      Distance between the two points
  """
  return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

async def search_places(
  api_key: str,
  query: str,
  page: int = 1,
  size: int = 5,
  ref_wtm_x: float | None = None,
  ref_wtm_y: float | None = None,
  nearest_only: bool = False
) -> KakaoSearchResponse:
  """
  Search for places using Kakao Local API.

  Args:
      api_key: Kakao REST API key
      query: Search query text
      page: Page number (default: 1)
      size: Number of results per page (default: 5, max: 45)
      ref_wtm_x: Reference X coordinate in WTM format for distance calculation
      ref_wtm_y: Reference Y coordinate in WTM format for distance calculation
      nearest_only: If True, returns only the nearest place to the reference coordinates

  Returns:
      KakaoSearchResponse containing search results and metadata

  Raises:
      httpx.HTTPError: If the API request fails
  """
  from rich.console import Console
  console = Console()
  
  encoded_query = urllib.parse.quote(query)
  url = f"https://dapi.kakao.com/v2/local/search/keyword.json?query={encoded_query}&page={page}&size={size}"
  
  console.print(f"🔍 Searching Kakao API for: '{query}'", style="blue")
  console.print(f"🌐 URL: {url}", style="blue")

  headers = {
    "Authorization": f"KakaoAK {api_key}",
    "Accept": "*/*",
    "Accept-Language": "ko,en-US;q=0.9,en;q=0.8",
  }

  async with httpx.AsyncClient() as client:
    response = await client.get(url, headers=headers)
    response.raise_for_status()
    data = response.json()
    result = KakaoSearchResponse.model_validate(data)
    
    console.print(f"📊 Found {len(result.documents)} results for '{query}'", style="blue")
    
    # If reference coordinates are provided and nearest_only is True,
    # find the nearest place and return only that one
    if ref_wtm_x is not None and ref_wtm_y is not None and nearest_only and result.documents:
      console.print(f"🔄 Finding nearest place to coordinates: ({ref_wtm_x}, {ref_wtm_y})", style="blue")
      # Calculate distance for each place
      nearest_place = None
      min_distance = float('inf')
      
      for place in result.documents:
        console.print(f"🏢 Processing place: '{place.place_name}' at coordinates: ({place.x}, {place.y})", style="blue")
        place_wtm_x, place_wtm_y = await convert_coordinates(api_key, str(place.x), str(place.y))
        distance = calculate_distance(ref_wtm_x, ref_wtm_y, place_wtm_x, place_wtm_y)
        place.calculated_distance = distance
        console.print(f"📏 Distance: {distance:.2f} to '{place.place_name}'", style="blue")
        
        if distance < min_distance:
          min_distance = distance
          nearest_place = place
      
      # Create a new response with only the nearest place
      if nearest_place:
        console.print(f"✅ Nearest place: '{nearest_place.place_name}' at distance {nearest_place.calculated_distance:.2f}", style="green")
        result.documents = [nearest_place]
    
    return result

# Example usage:
# async def main():
#     api_key = "your_api_key_here"
#     results = await search_places(api_key, "이태원 맛집")
#     for place in results.documents:
#         print(f"{place.place_name}: {place.road_address_name}")