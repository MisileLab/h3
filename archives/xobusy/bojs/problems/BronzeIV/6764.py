a = [int(input()) for _ in range(4)]
if all(i < j for i, j in zip(a, a[1:])):
  print("Fish rising")
elif all(i > j for i, j in zip(a, a[1:])):
  print("Fish diving")
elif all(i == j for i, j in zip(a, a[1:])):
  print("Fish At Constant Depth")
else:
  print("No Fish")
