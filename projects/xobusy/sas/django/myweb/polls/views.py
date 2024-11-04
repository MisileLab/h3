from django.http import HttpResponse
from time import sleep

def index(request):
    print("client")
    sleep(5)
    return HttpResponse("Hello, World. 204")
