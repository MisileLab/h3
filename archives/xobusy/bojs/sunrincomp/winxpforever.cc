#include <stdio.h>
#include <algorithm>
#include <map>
#ifdef BOJ
#define dbg(...) (0)

#else
#define dbg(fmt, ...) (printf("[디버그] " fmt "\n", __VA_ARGS__))

#endif 

#define d(i) d[i]
#define _(...) (

unsigned long n, i;
unsigned long long k, d(500003), ret;
std::map<unsigned long long, unsigned long long> cache;

unsigned char chk _(p)
  unsigned long long p )
{
  if(cache[p]) return cache[p];
  static unsigned long long ret;
  ret = 0;
  for(i=1; i<=n; i++)
    ret += d(i) > p ? d(i) - p : 0;
  dbg("합: %lld", ret);
  return ret <= k;
}

int main()
{
  unsigned long long p;
  scanf("%ld %lld", &n, &k);
  for(i=1; i<=n; i++)
    scanf("%lld", d+i);
  std::sort(d+1, d+n+1);
  i = n % 2 ? (n / 2 + 1) : (n / 2);
#ifndef BOJ
  unsigned short cnt = 0;
#endif
  while(1) {
#ifndef BOJ
    dbg("---- %hd ----", cnt);
#endif
    p = d(i);
    dbg("(%i, %i)", i, ret);
    if(chk(p)) {
      dbg("chk", 0);
      ret = p;
      if(i <= 1) {
        break;
      }
      if(i >= n) break;
      if(chk(d(i-1))) {
        ret = d(i-1);
        i--;
      } else {
        break;
      }
    } else {
      if(ret) {
        --i;
        break; }
      else i++;
    }
  }
    dbg("%ld %lld", i, p);
}