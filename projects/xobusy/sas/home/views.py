from django.shortcuts import HttpResponse, render

# Create your views here.
def index(request):
  return HttpResponse(b'<div /><h1>a</h1><div />')

