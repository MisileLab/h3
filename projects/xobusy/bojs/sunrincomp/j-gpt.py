a = [1]
n = int(input())
i = 2
iteration = 0

while n != i:
    a.append(i)

    if iteration % 3 == 2:
        i -= 1
    else:
        if i + 2 <= n and (i + 1 not in a or iteration == 0):
            i += 2
        else:
            i += 1

    iteration += 1

a.extend([5, 1])

print(n)
print(*a)
