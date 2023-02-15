a = []

def get_median(data):
    centerIndex = len(data)//2
    return (data[centerIndex] + data[-centerIndex - 1])/2

while True:
    try:
        b = input()
    except EOFError:
        break
    if b == "0":
        break
    a.append(list(map(int, b.split(" ")[1:])))

for i, i2 in zip(a, range(1, len(a)+1)):
    print(f"Case {i2}: {get_median(i):.1f}")
