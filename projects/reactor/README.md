# Mekanism Reactor Monitoring System

A real-time monitoring and control system for Minecraft Mekanism fusion reactors with physical hardware interface.

## Overview

This project provides a complete solution for monitoring and controlling Minecraft Mekanism fusion reactors through a physical hardware interface. The system consists of multiple components working together to provide real-time data visualization and control capabilities.

## System Architecture

```
[ComputerCraft Computer] --WebSocket--> [FastAPI Server] --TCP Socket--> [ESP8266] --Serial--> [Arduino Mega]
```

### Data Flow
* **Upstream**: ComputerCraft → FastAPI Server → ESP8266 → Arduino Mega (reactor status information)
* **Downstream**: Arduino Mega → ESP8266 → FastAPI Server → ComputerCraft (control commands)
* **Update Cycle**: 1 second

### Communication Methods
* ComputerCraft ↔ FastAPI Server: WebSocket bidirectional communication
* FastAPI Server ↔ ESP8266: TCP Socket (binary data)
* ESP8266 ↔ Arduino Mega: UART Serial communication (9600 baud)

## Hardware Components

### Main Components
* **Main Controller**: Arduino Mega 2560
* **WiFi Module**: ESP8266 (NodeMCU or Wemos D1 Mini)
* **Display**: 4 WS2812B LED strips (10 LEDs each)
* **Audio**: 1 Piezo buzzer
* **Input**: 4 tactile switches (1 emergency stop + 3 coolant control)
* **Power**: 5V/3A DC adapter

### Physical Layout
```
[Temperature LED Bar] [Fuel LED Bar] [Coolant LED Bar] [Waste LED Bar] [Emergency Stop]
                                                                     [Coolant-Low]
                                                                     [Coolant-Medium]
                                                                     [Coolant-High]
```

## Software Components

### 1. ComputerCraft (Lua)
* Collects reactor status data
* Transmits real-time data via WebSocket
* Receives and executes control commands
* Uses CC: Tweaked's websocket API

### 2. FastAPI Server (Python)
* Provides WebSocket endpoint
* Bidirectional real-time communication with ComputerCraft
* TCP socket communication with ESP8266
* Binary data conversion
* Data logging and analysis with Polars

### 3. ESP8266 (C++ / Arduino IDE)
* WiFi connection management
* TCP socket client
* Serial communication with Arduino Mega
* Data relay and protocol conversion

### 4. Arduino Mega (C++ / Arduino IDE)
* WS2812B LED control (FastLED library)
* Button input processing (with debouncing)
* Buzzer control
* Serial communication with ESP8266

## Data Logging

### Logging System
* **Library**: Polars (high-performance DataFrame library)
* **Storage Format**: Parquet files (.parquet)
* **Storage Cycle**: Real-time storage every 1 second
* **File Management**: Daily file splitting (reactor_YYYY-MM-DD.parquet)

### Data Schema
```python
Schema = {
    "timestamp": pl.Datetime,       # Timestamp
    "temperature": pl.Float32,      # Reactor temperature
    "fuel_level": pl.Float32,       # Fuel level (%)
    "coolant_level": pl.Float32,    # Coolant level (%)
    "waste_level": pl.Float32,      # Nuclear waste level (%)
    "emergency_stop": pl.Boolean,   # Emergency stop status
    "coolant_speed": pl.UInt8,      # Coolant speed (0=stop, 1=low, 2=medium, 3=high)
    "alert_status": pl.UInt8        # Alert status (0=normal, 1=caution, 2=danger)
}
```

## Data Protocols

### TCP Socket Protocol (FastAPI ↔ ESP8266)
```
[Header:1byte][Temperature:2bytes][Fuel:2bytes][Coolant:2bytes][Waste:2bytes][Checksum:1byte]
```

### Serial Communication Protocol (ESP8266 ↔ Arduino)
* **Baud Rate**: 9600 bps
* **Data Format**: JSON-based text protocol
```json
{
  "temp": 75.5,
  "fuel": 80.2,
  "coolant": 45.8,
  "waste": 15.3
}
```

### Control Commands
```json
{
  "cmd": "emergency_stop"
}
```
```json
{
  "cmd": "coolant_speed",
  "value": 2
}
```

## API Endpoints

### WebSocket
* `/ws/computercraft/{computer_id}` - WebSocket endpoint for ComputerCraft communication

### HTTP
* `GET /status` - Get current system status
* `GET /data?limit={n}` - Get recent data log entries (default 100)

## Installation

1. Install dependencies:
   ```bash
   uv sync
   ```

2. Run the server:
   ```bash
   python main.py
   ```

3. The server will start on `ws://localhost:8765`

## Development

### Adding New Features

To extend the system with additional monitoring parameters:

1. Update the data schema in `main.py`
2. Modify the TCP packet structure in `send_to_esp8266()`
3. Update the Arduino firmware to handle new parameters
4. Add new visualization components to the hardware interface

### Testing

1. Run the server:
   ```bash
   python main.py
   ```

2. Check the server status:
   ```bash
   curl http://localhost:8765/status
   ```

## Hardware Advantages

### Arduino Mega 2560 Benefits
* **Abundant I/O pins**: 54 digital, 16 analog pins for excellent expandability
* **Multiple serial ports**: Serial1 dedicated for ESP8266 use
* **Stability**: Proven platform for long-term stable operation
* **Library support**: Extensive libraries like FastLED, SoftwareSerial

### ESP8266 Separation Benefits
* **Dedicated WiFi processing**: Network communication handled separately from main logic
* **Easy upgrades**: WiFi module can be replaced/upgraded independently
* **Debugging convenience**: Serial monitor for independent module debugging

## Future Expansion Plans

### Additional Features
* Web dashboard (Parquet data visualization)
* Historical data analysis and charts
* Multi-reactor monitoring (Arduino expansion)
* Mobile app integration
* Predictive analytics (ML-based anomaly detection)
* LCD display addition (real-time value display)
* Temperature sensors (hardware temperature monitoring)

## Cost Estimate

| Item | Quantity | Unit Price | Total |
|------|----------|------------|-------|
| Arduino Mega 2560 | 1 | ₩18,000 | ₩18,000 |
| ESP8266 (NodeMCU) | 1 | ₩8,500 | ₩8,500 |
| WS2812B LED Strip (1m/60LED) | 1 | ₩10,000 | ₩10,000 |
| Tactile Switches (4) | 4 | ₩350 | ₩1,400 |
| Piezo Buzzer | 1 | ₩1,500 | ₩1,500 |
| Jumper Wires, Breadboard | - | - | ₩4,000 |
| Resistors, Capacitors, etc. | - | - | ₩3,000 |
| 5V/3A DC Adapter | 1 | ₩7,000 | ₩7,000 |
| Case Materials (Acrylic/3D Printing) | - | - | ₩5,000 |

**Total Estimated Cost: ₩58,400**

