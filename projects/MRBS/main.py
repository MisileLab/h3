from simple_websocket_server import WebSocketServer, WebSocket

clients = []
class AEBSWebsocket(WebSocket):
    def handle(self):
        self.send_message(f'{self.data} from server')
        broadcast("asd")
        print(self.data)
    
    def connected(self):
        print(self.address, 'connected')
        clients.append(self)

    def handle_close(self):
        print(self.address, 'closed')

server = WebSocketServer('0.0.0.0', 25567, AEBSWebsocket)

def broadcast(msg: str):
    for client in clients:
        client.send_message(msg)

server.serve_forever()
