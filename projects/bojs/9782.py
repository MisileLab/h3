from sys import stdin

def get_median(data):
    data = sorted(data)
    centerIndex = len(data)//2
    return (data[centerIndex] + data[-centerIndex - 1])/2

a = []

for i in stdin:
    if i.strip("\n") in ["0", ""]:
        break
    a.append(list(map(int, i.strip("\n").split(" ")[1:])))

for i, i2 in zip(a, range(1, len(a)+1)):
    print(f"Case {i2}: {get_median(i):.1f}")
