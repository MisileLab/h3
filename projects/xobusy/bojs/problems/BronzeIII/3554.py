def simulate_machine(sequence, operations):
  result = []

  for op in operations:
    k, l, r = op
    if k == 1:
      for i in range(l - 1, r):
        sequence[i] = (sequence[i] * sequence[i]) % 2010
    elif k == 2:
      result.append(sum(sequence[l - 1:r]))

  return result


if __name__ == "__main__":
  n = int(input())
  sequence = list(map(int, input().split()))
  m = int(input())
  operations = []

  for _ in range(m):
    k, l, r = map(int, input().split())
    operations.append((k, l, r))

  output = simulate_machine(sequence, operations)

  for val in output:
    print(val)
