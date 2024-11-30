#include <stdio.h>
#include <algorithm>
#include <vector>
unsigned short n, i, d[1003];
std::vector<unsigned short> v1, v2;
unsigned long ret;
int main()
{
  scanf("%hd", &n);
  for(i=1; i<=n; i++)
    scanf("%hd", d+i);
  std::sort(d+1, d+n+1);
  for(i=1; i<=n; i++) {
    if(i % 2 || ((i < n && d[i] != d[i+1]))) v1.push_back(d[i]);
    else v2.push_back(d[i]);
  }
  std::reverse(v2.begin(), v2.end());
  for(i=0; i<v1.size(); i++)
    ret += v1[i];
  for(i=0; i<v2.size(); i++)
    ret += v2[i];
  printf("%ld\n", ret);
  for(i=0; i<v1.size(); i++)
    printf("%hd ", v1[i]);
  for(i=0; i<v2.size(); i++)
    printf("%hd ", v2[i]);
  return 0;
}