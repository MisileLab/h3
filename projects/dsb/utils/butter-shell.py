from gradio_client import Client
from os import environ

a = input("enter message: ")
print(Client("https://butter.misile.xyz", auth=(environ["INF_USER"], environ["INF_PASS"])).predict(content=a, user="misile", audio_path=None))

