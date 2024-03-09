def santa_claus(N):
  # 이동 횟수와 방문 순서를 저장할 리스트 초기화
  dp = [0] * (N + 1)
  visit = [0] * (N + 1)

  # 1번 집부터 시작하여 2번 집부터 N번 집까지 순회하며 최소 이동 횟수 계산
  for i in range(2, N + 1):
    dp[i] = dp[i - 2] + 1

  # 마지막 집에서 1번 집으로 돌아오는 경우를 고려하여 최소 이동 횟수 계산
  dp[N] = min(dp[N], dp[N - 1] + 1, dp[N - 2] + 1)

  # 방문 순서 저장
  visit[1] = 1
  visit[2] = 1

  for i in range(3, N + 1):
    # 이동 횟수가 짝수일 때는 시계 방향으로 이동
    visit[i] = visit[i - 2] if dp[i] % 2 == 0 else visit[i - 1]
  # 결과 출력
  print(dp[N])
  for i in range(1, N + 1):
    print(visit[i], end=" ")

# 입력 예시
N = int(input())
santa_claus(N)
