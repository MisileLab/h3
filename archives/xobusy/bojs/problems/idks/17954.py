def main():
  n = int(input())

  if n == 1:
    print(2)
    print(1)
    print(2)
    return

  seq = list(range(2 * n - 3, n - 2, -1))
  seq.extend((2 * n - 2, 2 * n - 1))
  seq.extend(iter(range(n - 2, 0, -1)))
  seq.append(2 * n)

  sum_val = 1 * n * (2 * n + 1)
  ans = 0
  for i in range(2 * n):
    sum_val -= seq[i]
    ans += (i + 1) * sum_val

  print(ans)

  # Print the arrangement
  print(2 * n, end=' ')
  for i in range(1, n):
    print(i, end=' ')
  print(2 * n - 1)


if __name__ == "__main__":
  main()
