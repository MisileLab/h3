a = "9780921418"
b = 0
c = True

for _ in range(3):
    a = a + input()    

for i in a:
    i = int(i)
    if c:
        b += i
    else:
        b += i * 3
    c = not c
 
print(f"The 1-3-sum is {b}")
