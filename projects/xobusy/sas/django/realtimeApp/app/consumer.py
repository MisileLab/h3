from collections import defaultdict
from channels.generic.websocket import WebsocketConsumer
from argon2 import PasswordHasher

websockets = defaultdict(list)

class MessageConsumer(WebsocketConsumer):
  def connect(self):
    self.accept()
    self.name = ""
    self.verified = False

  def disconnect(self, code):
    if self.name != "":
      websockets[self.name].remove(self)

  def receive(self, text_data=None, bytes_data=None):
    from .models import Room
    if text_data is None:
      return
    if self.name == "":
      self.name = text_data
    elif not self.verified:
      try:
        PasswordHasher().verify(Room.objects.filter(name=self.name)[0].password, text_data) # type: ignore
      except Exception as e:
        print(e)
        self.disconnect(400)
      else:
        websockets[self.name].append(self)
        self.verified = True
