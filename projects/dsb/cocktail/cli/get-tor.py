from httpx import get

from platform import system

r = input("are you not x86_64, then please input [y] and if you are not input [n]: ")
while not r in ["y","n"]:
 r = input("Invaild answer: ")
if r == "y":
 print("ok, plz install tor to ./tor")
else:
 print("ok, you are x86_64")
 if system() == "linux":
  print("linux has own tor package, plz install that and link to ./tor")
  exit()
 else:
  print("let's ping torproject")
  r = get("https://dist.torproject.org/torbrowser")
  if r.is_error:
   print("i think tor banned from your network, let's try email thing")
   print("1. send email with your system(windows, linux, mac) at title to gettor@torproject.org")
   print("2. go to https://dist.torproject.org/torbrowser with tor browser and executable to ./tor")
  else:
   print("go to https://dist.torproject.org/torbrowser and get a file and move to ./tor")

