#include <iostream>
#include <vector>
#include <algorithm>

using namespace std;
int main() {
  std::vector<int> a {1};
  int n;
  std::cin >> n;
  int i = 2;
  int np = 1;

  while (n != i) {
    a.push_back(i);

    if (np == 2) {
      i -= 1;
    } else {
      i += (i + 2 <= n && (std::find(a.begin(), a.end(), i + 1) == a.end() && np != 1)) ? 1 : 2;
    }

    np += 1;
    if (np == 3) {
      np = 0;
    }
  }

  a.push_back(5);
  a.push_back(1);

  std::cout << n << std::endl;
  for (int num : a) {
    std::cout << num << " ";
  }
  std::cout << std::endl;

  return 0;
}
