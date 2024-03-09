a = int(input())
strings = [
  " @@@   @@@  ",
  "@   @ @   @ ",
  "@  @  @ ",
  "@     @ ",
  " @     @  ",
  "  @   @   ",
  "   @   @  ",
  "  @ @   ",
  "   @    "
]
print('\n'.join([(i * a) for i in strings]))
