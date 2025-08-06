import os
import pytest
from dotenv import load_dotenv

from src.core.kakao_api import search_places, KakaoSearchResponse

# Load environment variables
load_dotenv()

@pytest.mark.asyncio
async def test_search_places():
  """Test the Kakao Local API search functionality."""
  api_key = os.getenv("KAKAO_API_KEY")
  if not api_key:
    pytest.skip("KAKAO_API_KEY environment variable not set")

  # Test basic search
  response = await search_places(api_key, "이태원 맛집")
  assert isinstance(response, KakaoSearchResponse)
  assert len(response.documents) > 0
  
  # Test pagination
  response_page_2 = await search_places(api_key, "이태원 맛집", page=2)
  assert isinstance(response_page_2, KakaoSearchResponse)
  assert len(response_page_2.documents) > 0
  
  # Verify different results between pages
  first_ids = {place.id for place in response.documents}
  second_ids = {place.id for place in response_page_2.documents}
  assert first_ids != second_ids, "Page 1 and 2 should have different results"

  # Test size parameter
  response_small = await search_places(api_key, "이태원 맛집", size=5)
  assert len(response_small.documents) == 5

  # Test metadata
  assert response.meta.total_count > 0
  assert isinstance(response.meta.is_end, bool)
  assert response.meta.pageable_count > 0

  # Test place fields
  first_place = response.documents[0]
  assert first_place.place_name
  assert first_place.address_name
  assert first_place.road_address_name
  assert first_place.x  # longitude
  assert first_place.y  # latitude