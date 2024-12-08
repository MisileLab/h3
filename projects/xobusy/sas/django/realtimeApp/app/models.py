from django.db import models

class Message(models.Model):
  content = models.TextField()
  room = models.TextField()
