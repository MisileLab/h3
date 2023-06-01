import math

def lcm(a: int, b: int) -> int:
    return abs(a * b) // math.gcd(a, b)

def find_largest_lcm(n: int) -> int:
    answer = 0
    for i in range(1, n - 1):
        temp = lcm(i, lcm(i + 1, i + 2))
        answer = max(answer, temp)
    return answer

num_cases = int(input(""))
for _ in range(num_cases):
    n = int(input(""))
    largest_lcm = find_largest_lcm(n)
    print(largest_lcm)
