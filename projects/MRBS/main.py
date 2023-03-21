from simple_websocket_server import WebSocketServer, WebSocket
from json import loads, dumps

clients = []
answers = []

def broadcast(self, msg: str):
    for client in clients:
        if self != client:
            client.send_message(msg)

class AEBSWebsocket(WebSocket):
    def handle(self):
        if self.data in ["T", "TF", "A", "AS", "I"]:
            broadcast(self, dumps({"name": self.data}))
        elif self.data == "ID":
            try:
                data = loads(self.data.split(" ")[1])
                data["name"] = "ID"
            except Exception:
                self.send_message(dumps({"name": "IC"}))
            else:
                broadcast(self, dumps(data))
        elif self.data[0] == "B" and self.data.removeprefix("B").isnumeric():
            broadcast(self, dumps({"name": "B", "rate": int(self.data.removeprefix("B"))}))
        print(f"{self.address}: {self.data}")

    def connected(self):
        print(self.address, 'connected')
        clients.append(self)

    def handle_close(self):
        print(self.address, 'closed')
        clients.remove(self)

server = WebSocketServer('0.0.0.0', 25567, AEBSWebsocket)
server.serve_forever()
