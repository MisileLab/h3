<script lang="ts">
  import { Presentation, Slide, Code } from '@animotion/core'

  let code: Code;
  const codes = [
    `def receive(self, text_data=None, bytes_data=None):
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
        self.verified = True`,
    `from django.db import models
    class Message(models.Model):
      content = models.TextField()
      room = models.TextField()

    class Room(models.Model):
      name = models.TextField()
      password = models.TextField()
    `,
    `from .consumer import MessageConsumer
    urlpatterns = [path("ws", MessageConsumer.as_asgi())]`,
    `from django.shortcuts import redirect, render
    from argon2 import PasswordHasher
    from .models import Message, Room
    from .consumer import websockets

    def send(request):
      text = request.GET.get('text', '')
      room = request.GET.get('room', '')
      password = request.GET.get('password', '')
      if room == '':
        return redirect('/')
      roomobj = Room.objects.filter(name=room) # type: ignore`,
    `if not roomobj:
        roomobj = Room(name=room, password=PasswordHasher().hash(password))
        roomobj.save()
      else:
        roomobj = roomobj[0]
      try:
        PasswordHasher().verify(str(roomobj.password), password)
      except Exception as e:
        return redirect('/')
      if text != '':
        Message(room=room, content=text).save()
        for i in websockets[room]:
          i.send(text)
      messages = [i.content for i in Message.objects.filter(room=room)]
      return render(request, "send.html",` + '{"room": room, "messages": messages, "password": password})'
  ];
</script>

<Presentation options={{history: true, transition: 'none', progress: false, controls: false}}>
  <Slide class="h-full place-content-center place-items-center">
    <h1 class="text-8xl">Realtime chat app</h1>
  </Slide>
  <Slide class="h-full place-content-center place-items-center">
    <h1 class="text-8xl">시연</h1>
  </Slide>
  <Slide class="h-full place-content-center place-items-center text-5xl gap-y-6">
    <h1 class="text-8xl">코드</h1>
    <Code lang="python" codes={codes} bind:this={code} />
  </Slide>
</Presentation>
