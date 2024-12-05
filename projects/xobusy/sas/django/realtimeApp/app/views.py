from django.shortcuts import render
from django.http import JsonResponse
from Crypto.PublicKey import ECC
from zstandard import ZstdDecompressor as zstd

from models import Chat, User

from base64 import b64decode

def index(request):
  return render(request, "index.html")

def register_pk(request):
  pk = request.GET.get('pk', '')
  if pk == '':
    return JsonResponse({'status': 'failed'})
  data = zstd().decompress(b64decode(pk)).decode()
