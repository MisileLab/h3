from z3 import Int, Solver
import lib as mrl

lst = bytes.fromhex("AC F3 0C 25 A3 10 B7 25 16 C6 B7 BC 07 25 02 D5 C6 11 07 C5 00 00 00 00 00 00 00 00 00 00 00 00")
x = [Int(str(i)) for i in range(21)]
s = Solver()
mrl.add_ascii(s, x)
for i in range(18):
  s.insert((x[i] + i) * -5 == lst[i])

s.check()
print(s.model())
print(mrl.print_flag(s, x))
