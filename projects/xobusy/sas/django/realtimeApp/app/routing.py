from django.urls import path
from .consumer import MessageConsumer

urlpatterns = [path("ws", MessageConsumer.as_asgi())]
