from django.shortcuts import render
from dust.api import check_air

def index(request):
  return check_air(name='Yongsan')

def detail(request):
  res = check_air(name="Seoul")


