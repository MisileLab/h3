from channels.generic.websocket import WebsocketConsumer

websockets: dict[str, list[WebsocketConsumer]] = {}

class MessageConsumer(WebsocketConsumer):
  def connect(self):
    self.accept()
    self.name = ""

  def disconnect(self, code):
    if self.name != "":
      websockets[self.name].remove(code)

  def receive(self, text_data=None, bytes_data=None):
    if text_data is None:
      return
    if self.name == "":
      self.name = text_data
