def div(i: list) -> int:
  return i[0] / i[1]

a = [div(list(map(int, input().split())))]
a.extend(div(i) for i in [list(map(int, input().split())) for _ in range(int(input()))])

print(f"{min(a)*1000:.2f}")
