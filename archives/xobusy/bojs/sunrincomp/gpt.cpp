#include <iostream>
#include <vector>
using namespace std;

int main() {
  int N, M;
  cin >> N;
  cin >> M;

  // 최대 사용 가능한 색상 수
  int maxColors = N * M;

  // 이차원 배열 생성
  vector<vector<int>> arr(M, vector<int>(N));

  // 배열 채우기
  int color = 1;
  for (int i = 0; i < M; i++) {
    for (int j = 0; j < N; j++) {
      arr[i][j] = color;
      color = (color % maxColors) + 1;
    }
  }

  // 배열 출력
  cout << "이차원 배열:\n";
  for (int i = 0; i < M; i++) {
    for (int j = 0; j < N; j++) {
      cout << arr[i][j] << " ";
    }
    cout << endl;
  }

  cout << "최소 색상의 개수: " << maxColors << endl;

  return 0;
}
