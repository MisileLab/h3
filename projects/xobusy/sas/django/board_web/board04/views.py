from django.shortcuts import render
from django.http import HttpResponse

def index(r):
  print("Received")
  return HttpResponse(b'<div></div>')

# Create your views here.
