from django.db import models

class Chat(models.Model):
  content = models.TextField()
  sender = models.TextField()
  receiver = models.TextField()

class User(models.Model):
  name = models.TextField()
  pk = models.TextField()
