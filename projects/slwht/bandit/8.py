with open('data.txt') as f:
    b = f.read().split("\n")
    for i in b:
        if b.count(i) == 1:
            print(i)
            break
