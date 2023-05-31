from functools import reduce
# https://www.w3resource.com/python-exercises/basic/python-basic-1-exercise-135.php
def test(nums):
    return reduce(lambda x,y:lcm(x,y),nums)
def gcd(a, b):
    while b:
        a, b = b, a%b
    return a
def lcm(a, b):
    return a*b // gcd(a, b)

for i in [int(input()) for _ in range(int(input()))]:
    numbers = list(range(1, i+1))
    answern = 0
    for i2 in range(len(numbers) - 2):
        _temp = test([numbers[i2], numbers[i2+1], numbers[i2+2]])
        if max(_temp, answern) == _temp:
            answern = _temp
    print(answern)
