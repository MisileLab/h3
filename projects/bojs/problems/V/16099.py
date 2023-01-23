a = []

for _ in range(int(input())):
    a.append(list(map(int, input().split())))

for i in a:
    if i[0] * i[1] == i[2] * i[3]:
        print('Tie')
    elif i[0] * i[1] < i[2] * i[3]:
        print('Eurecom')
    else:
        print('TelecomParisTech')

