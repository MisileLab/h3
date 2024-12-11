from django.shortcuts import redirect, render
from argon2 import PasswordHasher
from .models import Message, Room
from .consumer import websockets

def index(request):
  return render(request, "index.html")

def send(request):
  text = request.GET.get('text', '')
  room = request.GET.get('room', '')
  password = request.GET.get('password', '')
  if room == '':
    return redirect('/')
  roomobj = Room.objects.filter(name=room) # type: ignore
  if not roomobj:
    roomobj = Room(name=room, password=PasswordHasher().hash(password))
    roomobj.save()
  else:
    roomobj = roomobj[0]
  try:
    PasswordHasher().verify(str(roomobj.password), password)
  except Exception as e:
    print(e)
    return redirect('/')
  if text != '':
    Message(room=room, content=text).save()
    for i in websockets[room]:
      i.send(text)
  messages = []
  for i in Message.objects.filter(room=room): # type: ignore
    messages.append(i.content)
  return render(request, "send.html", {"room": room, "messages": messages, "password": password})
