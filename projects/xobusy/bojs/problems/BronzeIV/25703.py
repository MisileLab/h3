a = int(input())
print("int a;")
for i in range(1, a+1):
  if i == 1:
    print("int *ptr = &a;")
  elif i == 2:
    print("int **ptr2 = &ptr;")
  else:
    print(f"int {'*' * i}ptr{i} = &ptr{i-1};")
