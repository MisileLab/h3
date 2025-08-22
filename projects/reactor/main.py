import asyncio

from websockets.asyncio.server import ServerConnection, serve

async def handler(websocket: ServerConnection):
  async for message in websocket:
    print(f"Received message: {message}")
    await websocket.send(f"Echo: {message}")

async def main():
  async with serve(handler, "localhost", 8765) as server:
    print("WebSocket server started on ws://localhost:8765")
    await server.serve_forever()

asyncio.run(main())

