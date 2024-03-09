#include <stdio.h>
#include <algorithm>
unsigned short n, i, j, ret;
unsigned long w[5003], k, x;
unsigned char chk[5003];
int main()
{
  scanf("%hd %ld", &n, &k);
  for(i=1; i<=n; i++)
    scanf("%ld", w+i);
  std::sort(w+1, w+n+1, [](unsigned long a, unsigned long b) { return a > b; });
  for(i=1; i<=n; i++) {
    if(w[i] == k) continue;
    if(chk[i]) continue;
    x = 0;
    for(j=n; j>i; j--) {
      if(chk[j]) continue;
      if(w[i] + w[j] <= k)
        x = w[i] + w[j];
      else
        break;
    }
    if(x) ret++, chk[j] = 1, chk[i] = 1, printf("%ld %ld\n", w[i],w[j]);
  }
  printf("%hd", ret);
}