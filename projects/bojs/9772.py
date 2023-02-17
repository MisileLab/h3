a = []

while True:
    b, c = map(float, input().split(" "))
    a.append((b, c))
    if b == 0 and c == 0:
        break

for i in a:
    b, c = i
    if b == 0 or c == 0:
        print("AXIS")
    elif b < 0:
        if c > 0:
            print("Q2")
        else:
            print("Q3")
    elif c > 0:
        print("Q1")
    else:
        print("Q4")
