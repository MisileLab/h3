import heapq

N, M = map(int, input().split())

# 그래프 생성
graph = [[] for _ in range(N + 1)]
in_degree = [0] * (N + 1)

for _ in range(M):
  v, w = map(int, input().split())
  graph[w].append(v)
  in_degree[v] += 1

# 위상 정렬
queue = []
for i in range(1, N + 1):
  if in_degree[i] == 0:
    heapq.heappush(queue, i)

B = [0] * N  # 재배열된 수열을 저장할 리스트
index = 0  # 수열의 순서로 값 배치

while queue:
  u = heapq.heappop(queue)
  B[index] = u
  index += 1

  for v in graph[u]:
    in_degree[v] -= 1
    if in_degree[v] == 0:
      heapq.heappush(queue, v)

# 결과 출력
max_value = sum(B) * 0.5  # 수열의 가치의 최댓값은 합의 절반
print(max_value)
print(' '.join(map(str, B)))
