import asyncio
import json
import logging
import struct
from typing import Dict, Any
from datetime import datetime

import polars as pl
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
import uvicorn
import socketio

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(title="Reactor Monitoring System", version="1.0.0")

# Initialize SocketIO server for ESP8266 communication
sio = socketio.AsyncServer(async_mode='asgi')
combined_app = socketio.ASGIApp(sio, app)

# Store connections
class ConnectionManager:
    def __init__(self):
        self.computercraft_connections: Dict[str, WebSocket] = {}
        self.esp8266_connected = False
        self.reactor_data: Dict[str, Any] = {
            "temperature": 0.0,
            "fuel_level": 0.0,
            "coolant_level": 0.0,
            "waste_level": 0.0,
            "status": False,
            "burn_rate": 0.0,
            "actual_burn_rate": 0.0,
            "alert_status": 0
        }
        self.data_buffer = []

# Initialize connection manager
manager = ConnectionManager()

# Data logging setup
def initialize_data_logging():
    """Initialize the data logging with Polars"""
    schema = {
        "timestamp": pl.Datetime,
        "temperature": pl.Float32,
        "fuel_level": pl.Float32,
        "coolant_level": pl.Float32,
        "waste_level": pl.Float32,
        "status": pl.Boolean,
        "burn_rate": pl.Float32,
        "actual_burn_rate": pl.Float32,
        "alert_status": pl.UInt8
    }
    return pl.DataFrame(schema=schema)

# Initialize data log
data_log = initialize_data_logging()

@app.websocket("/ws/computercraft/{computer_id}")
async def websocket_endpoint(websocket: WebSocket, computer_id: str):
    """Handle WebSocket connections from ComputerCraft"""
    await websocket.accept()
    manager.computercraft_connections[computer_id] = websocket
    logger.info(f"ComputerCraft {computer_id} connected")

    try:
        while True:
            # Receive data from ComputerCraft
            data = await websocket.receive_text()
            logger.info(f"Received from ComputerCraft {computer_id}: {data}")

            # Parse the data
            try:
                reactor_data = json.loads(data)
                # Update stored reactor data
                manager.reactor_data.update(reactor_data)

                # Log data with timestamp
                log_entry = reactor_data.copy()
                log_entry["timestamp"] = datetime.now()
                manager.data_buffer.append(log_entry)

                # Send data to ESP8266 if connected
                if manager.esp8266_connected:
                    await send_to_esp8266(reactor_data)

            except json.JSONDecodeError:
                logger.error("Failed to parse JSON from ComputerCraft")
                continue

    except WebSocketDisconnect:
        logger.info(f"ComputerCraft {computer_id} disconnected")
        if computer_id in manager.computercraft_connections:
            del manager.computercraft_connections[computer_id]

async def send_to_esp8266(data: Dict[str, Any]):
    """Send data to ESP8266 via SocketIO"""
    try:
        # Convert data to binary format for efficient transmission
        # Format: [Header:1byte][Temperature:2bytes][Fuel:2bytes][Coolant:2bytes][Waste:2bytes][Status:1byte][Alert:1byte][Checksum:1byte]
        temp = int(data.get("temperature", 0) * 10)  # Convert to integer with 0.1 precision
        fuel = int(data.get("fuel_level", 0) * 10)
        coolant = int(data.get("coolant_level", 0) * 10)
        waste = int(data.get("waste_level", 0) * 10)
        status = 1 if data.get("status", False) else 0
        alert = data.get("alert_status", 0)

        # Create binary packet
        packet = struct.pack(
            "!BHHHHBB",
            0xAA,  # Header
            min(temp, 65535),  # Temperature (0-6553.5)
            min(fuel, 65535),  # Fuel level (0-6553.5)
            min(coolant, 65535),  # Coolant level (0-6553.5)
            min(waste, 65535),  # Waste level (0-6553.5)
            status,  # Status (0/1)
            alert,   # Alert status (0-2)
            0x55   # Checksum (simplified)
        )

        # Emit to ESP8266
        await sio.emit('reactor_data', {'data': packet.hex()})
        logger.info("Sent data to ESP8266")

    except Exception as e:
        logger.error(f"Failed to send data to ESP8266: {e}")

@sio.event
async def connect(sid, environ):
    """Handle ESP8266 connection"""
    logger.info(f"ESP8266 connected with sid: {sid}")
    manager.esp8266_connected = True

@sio.event
async def disconnect(sid):
    """Handle ESP8266 disconnection"""
    logger.info(f"ESP8266 disconnected with sid: {sid}")
    manager.esp8266_connected = False

@sio.event
async def control_command(sid, data):
    """Handle control commands from ESP8266"""
    logger.info(f"Received control command from ESP8266: {data}")

    # Forward command to all connected ComputerCraft clients
    command_data = json.dumps(data)
    for computer_id, websocket in manager.computercraft_connections.items():
        try:
            await websocket.send_text(command_data)
        except Exception as e:
            logger.error(f"Failed to send command to ComputerCraft {computer_id}: {e}")

@app.get("/status")
async def get_status():
    """Get current system status"""
    return {
        "computercraft_connections": list(manager.computercraft_connections.keys()),
        "esp8266_connected": manager.esp8266_connected,
        "reactor_data": manager.reactor_data
    }

@app.get("/data")
async def get_data_log(limit: int = 100):
    """Get recent data log entries"""
    global data_log
    
    # Process buffered data
    if manager.data_buffer:
        new_data = pl.DataFrame(manager.data_buffer)
        data_log = pl.concat([data_log, new_data])
        manager.data_buffer.clear()

    # Return recent entries
    return data_log.tail(limit).to_dict(as_series=False)

if __name__ == "__main__":
    logger.info("Starting Reactor Monitoring Server...")
    uvicorn.run(combined_app, host="0.0.0.0", port=8765)

