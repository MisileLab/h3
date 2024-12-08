from channels.routing import ProtocolTypeRouter, URLRouter
from django.urls import path
from ..app.consumer import MessageConsumer

application = ProtocolTypeRouter({'websockets': URLRouter([path("ws", MessageConsumer.as_asgi())])})
