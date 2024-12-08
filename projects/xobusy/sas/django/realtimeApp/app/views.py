from django.shortcuts import redirect, render
from .models import Message
from .consumer import websockets

def index(request):
  return render(request, "index.html")

def send(request):
  text = request.GET.get('text', '')
  room = request.GET.get('room', '')
  if room == '':
    return redirect('/')
  if text != '':
    Message(room=room, content=text).save()
    for i in websockets[room]:
      i.send(text)
  messages = []
  for i in Message.objects.filter(room=room): # type: ignore
    messages.append(i.content)
  return render(request, "send.html", {"room": room, "messages": messages})
