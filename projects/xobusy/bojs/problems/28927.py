t, e, f = map(int, input().split())
t2, e2, f2 = map(int, input().split())
_max = t*3+e*20+f*120
mel = t2*3+e2*20+f2*120

if mel == _max:
  print("Draw")
elif mel > _max:
  print("Mel")
else:
  print("Max")
