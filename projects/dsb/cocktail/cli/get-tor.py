from httpx import get

from platform import system

print("ok, you are x86_64")
if system() == "linux":
 print("linux has own tor package, plz install that")
 exit()
else:
 print("let's ping torproject")
 try:
  r = get("https://torproject.org")
 except Exception as e:
  print(e)
  print("i think tor banned from your network, let's try email thing")
  print("1. send email with your system(windows, linux, mac) at title to gettor@torproject.org")
  print("2. go to https://torproject.org with tor browser and install tor service")
 else:
  print("go to https://torproject.org and get a file and install tor service")

