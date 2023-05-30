a, b, c = map(int, input().split(" "))
if c-(a-b) >= 0:
    print('Bus')
elif c-(a-b) == 0:
    print('Anything')
else:
    print('Subway')
