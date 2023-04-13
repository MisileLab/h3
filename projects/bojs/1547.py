a = int(input())
b = []

while True:
    try:
        c = map(int, input().split(" "))
    except EOFError:
        break
    
