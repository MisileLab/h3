# Reactor Monitoring System

A real-time monitoring system for Minecraft ComputerCraft fission reactors using FastAPI backend and Lua data collection.

## Architecture

The system consists of three main components:

1. **Python Backend** (`main.py`) - FastAPI server with WebSocket support
2. **Lua Data Collector** (`reactor_monitor.lua`) - ComputerCraft script for reactor monitoring
3. **ESP8266 Interface** - Optional hardware interface via SocketIO

## Data Structure Alignment

### Data Fields Sent by Lua Script

The Lua script (`reactor_monitor.lua`) collects and sends the following reactor data:

```lua
{
  "temperature": number,        -- Reactor temperature in Kelvin
  "fuel_level": number,         -- Fuel level as percentage (0-100)
  "coolant_level": number,      -- Coolant level as percentage (0-100)
  "waste_level": number,        -- Waste level as percentage (0-100)
  "status": boolean,            -- Reactor operational status
  "burn_rate": number,          -- Current burn rate
  "actual_burn_rate": number,   -- Actual burn rate
  "alert_status": number        -- Alert level (0=normal, 1=caution, 2=danger)
}
```

### Data Fields Expected by Python Backend

The Python backend (`main.py`) expects and processes the exact same data structure:

```python
{
    "temperature": 0.0,         # Float32
    "fuel_level": 0.0,          # Float32 (0-100)
    "coolant_level": 0.0,       # Float32 (0-100)
    "waste_level": 0.0,         # Float32 (0-100)
    "status": False,             # Boolean
    "burn_rate": 0.0,           # Float32
    "actual_burn_rate": 0.0,    # Float32
    "alert_status": 0            # UInt8 (0-2)
}
```

## Data Flow

1. **Lua Script** → Collects reactor data every second
2. **WebSocket** → Sends JSON data to Python backend
3. **Python Backend** → Processes and stores data, forwards to ESP8266 if connected
4. **ESP8266** → Receives binary-encoded data for hardware display

## Binary Data Format (ESP8266)

The backend converts reactor data to a binary format for efficient transmission:

```
[Header:1byte][Temperature:2bytes][Fuel:2bytes][Coolant:2bytes][Waste:2bytes][Status:1byte][Alert:1byte][Checksum:1byte]
```

- **Header**: 0xAA (start marker)
- **Temperature**: 16-bit integer (0.1°K precision)
- **Fuel Level**: 16-bit integer (0.1% precision)
- **Coolant Level**: 16-bit integer (0.1% precision)
- **Waste Level**: 16-bit integer (0.1% precision)
- **Status**: 8-bit boolean (0/1)
- **Alert Status**: 8-bit integer (0-2)
- **Checksum**: 0x55 (end marker)

## Alert System

The alert status is calculated based on reactor conditions:

- **0 (Normal)**: Temperature < 600°K, Coolant > 20%
- **1 (Caution)**: Temperature 600-1000°K OR Coolant < 20%
- **2 (Danger)**: Temperature > 1000°K

## Installation & Usage

### Backend Setup

1. Install dependencies:
   ```bash
   pip install fastapi uvicorn python-socketio polars
   ```

2. Run the server:
   ```bash
   python main.py
   ```

The server runs on `http://localhost:8765`

### ComputerCraft Setup

1. Place `reactor_monitor.lua` in your ComputerCraft computer
2. Ensure a fission reactor is connected via `fissionReactorLogicAdapter_0`
3. Run the script: `reactor_monitor`

## API Endpoints

- **WebSocket**: `/ws/computercraft/{computer_id}` - Real-time reactor data
- **GET** `/status` - Current system status
- **GET** `/data?limit=100` - Recent data log entries

## Error Handling

Both programs include comprehensive error handling:
- JSON parsing errors
- WebSocket connection failures
- Reactor component availability checks
- Graceful shutdown procedures

## Performance

- **Update Rate**: 1 second intervals
- **Data Storage**: Polars DataFrame for efficient time-series data
- **Binary Transmission**: Optimized for ESP8266 communication
- **Memory Management**: Automatic cleanup of disconnected clients

