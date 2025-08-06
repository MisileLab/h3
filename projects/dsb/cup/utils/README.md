# Utility Scripts

## Address to WTM Converter

This utility converts addresses to WTM (Web Mercator Transform) coordinates using the Kakao Maps API.

### Prerequisites

1. Set up your environment:
   ```bash
   # Create a .env file with your Kakao API key
   echo "KAKAO_API_KEY=your_api_key_here" > .env
   ```

2. Install required dependencies:
   ```bash
   uv add httpx python-dotenv
   ```

### Usage

The script can be used in three different ways:

#### 1. Convert a single address from the command line:

```bash
uv run python utils/address_to_wtm.py "서울특별시 강남구 테헤란로 152"
```

#### 2. Convert multiple addresses from a file:

```bash
uv run python utils/address_to_wtm.py -f addresses.txt -o results.csv
```

Where `addresses.txt` contains one address per line:
```
서울특별시 강남구 테헤란로 152
서울특별시 용산구 이태원동 34-149
부산광역시 해운대구 우동 센텀2로 25
```

The results will be saved to `results.csv` in CSV format.

#### 3. Interactive mode:

```bash
uv run python utils/address_to_wtm.py
```

This will start an interactive prompt where you can enter addresses one by one.
Type `exit` or press Ctrl+C to quit.

### Example Output

```
서울특별시 강남구 테헤란로 152: 1031533.6352, 1949860.5733
```

The output shows the address followed by the WTM X and Y coordinates.