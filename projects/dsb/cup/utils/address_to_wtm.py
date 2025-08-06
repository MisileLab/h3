#!/usr/bin/env python
import os
import sys
import asyncio
import argparse
from pathlib import Path
from dotenv import load_dotenv

# Add the parent directory to Python path
sys.path.append(str(Path(__file__).parent.parent))

from src.core.kakao_api import convert_coordinates

async def address_to_wtm(api_key: str, address: str) -> tuple[float, float]:
  """
  Convert an address to WTM coordinates.
  
  Args:
      api_key: Kakao API key
      address: Address to convert
      
  Returns:
      Tuple of (wtm_x, wtm_y) coordinates
  """
  import httpx
  
  # First, search for the address to get its coordinates
  url = "https://dapi.kakao.com/v2/local/search/address.json"
  params = {"query": address}
  headers = {"Authorization": f"KakaoAK {api_key}"}
  
  async with httpx.AsyncClient() as client:
    # Get the address coordinates
    response = await client.get(url, params=params, headers=headers)
    response.raise_for_status()
    data = response.json()
    
    if not data["documents"]:
      raise ValueError(f"Address not found: {address}")
    
    # Extract the coordinates
    x = data["documents"][0]["x"]
    y = data["documents"][0]["y"]
    
    # Convert to WTM
    wtm_x, wtm_y = await convert_coordinates(api_key, x, y, "WTM")
    
    return wtm_x, wtm_y

async def main():
  # Load environment variables
  load_dotenv()
  
  # Parse command line arguments
  parser = argparse.ArgumentParser(description="Convert an address to WTM coordinates")
  parser.add_argument("address", nargs="?", help="Address to convert")
  parser.add_argument("-f", "--file", help="File containing addresses (one per line)")
  parser.add_argument("-o", "--output", help="Output file for results")
  args = parser.parse_args()
  
  # Get API key
  api_key = os.getenv("KAKAO_API_KEY")
  if not api_key:
    print("Error: KAKAO_API_KEY environment variable not set")
    return 1
  
  # Process addresses
  results = []
  
  if args.file:
    # Process addresses from file
    try:
      with open(args.file, "r", encoding="utf-8") as f:
        addresses = [line.strip() for line in f if line.strip()]
      
      print(f"Converting {len(addresses)} addresses...")
      for address in addresses:
        try:
          wtm_x, wtm_y = await address_to_wtm(api_key, address)
          results.append((address, wtm_x, wtm_y))
          print(f"{address}: {wtm_x}, {wtm_y}")
        except Exception as e:
          print(f"Error converting '{address}': {e}")
          results.append((address, None, None))
    
    except Exception as e:
      print(f"Error reading file: {e}")
      return 1
  
  elif args.address:
    # Process single address from command line
    try:
      wtm_x, wtm_y = await address_to_wtm(api_key, args.address)
      results.append((args.address, wtm_x, wtm_y))
      print(f"{args.address}: {wtm_x}, {wtm_y}")
    except Exception as e:
      print(f"Error: {e}")
      return 1
  
  else:
    # Interactive mode
    print("Enter addresses to convert (press Ctrl+C or type 'exit' to quit):")
    try:
      while True:
        address = input("> ")
        if address.lower() == "exit":
          break
        
        if not address.strip():
          continue
        
        try:
          wtm_x, wtm_y = await address_to_wtm(api_key, address)
          results.append((address, wtm_x, wtm_y))
          print(f"WTM coordinates: {wtm_x}, {wtm_y}")
        except Exception as e:
          print(f"Error: {e}")
    
    except KeyboardInterrupt:
      print("\nExiting...")
  
  # Save results if output file specified
  if args.output and results:
    try:
      with open(args.output, "w", encoding="utf-8") as f:
        f.write("Address,WTM_X,WTM_Y\n")
        for address, x, y in results:
          if x is not None and y is not None:
            f.write(f"{address},{x},{y}\n")
          else:
            f.write(f"{address},ERROR,ERROR\n")
      print(f"Results saved to {args.output}")
    except Exception as e:
      print(f"Error saving results: {e}")
      return 1
  
  return 0

if __name__ == "__main__":
  sys.exit(asyncio.run(main()))