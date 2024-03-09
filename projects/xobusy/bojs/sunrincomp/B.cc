#include <stdio.h>
unsigned char d[11];
unsigned short t, ret;
unsigned char i, j;
unsigned char dd[] = { 0, 31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31 }, di;
char s[7];
void f()
{
  ret = 0;
  for(i=0; i<10; i++)
    scanf("%hhd", d+i);
  for(i=1; i<=12; i++) {
    di = dd[i];
    for(j=1; j<=di; j++) {
      sprintf(s, "%02hhd%02hhd", i, j);
      if(s[0] == 48) s[0] = 127;
      if(s[2] == 48) s[2] = 127;
      
      ret += (!d[s[0]-48] && !d[s[1]-48] && !d[s[2]-48] && !d[s[3]-48]);
      // puts(s);
    }
  }
  printf("%hd\n", ret);
}

int main()
{
  scanf("%hd", &t);
  while(t--)
    f();
  return 0;
}