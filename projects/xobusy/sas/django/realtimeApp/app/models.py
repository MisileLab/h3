from django.db import models

class Message(models.Model):
  content = models.TextField()
  room = models.TextField()

class Room(models.Model):
  name = models.TextField()
  password = models.TextField()
