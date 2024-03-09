#include <iostream>

using namespace std;
const int a = 20150115;
int n;
int cnt = 1;
int 목표 ;
int main(){
  cin >> n;
  목표 = n;
  cout << n << '\n';
  while(n){
    if(n>=4){
      n-=4;
      cout << cnt << " " << cnt +1 << " " << cnt+3 << " " << cnt+2 << " ";
      cnt += 4;
    }else if(n >= 3){
      n-=3;
      cout << cnt << " " << cnt +2 << " " << cnt+1 << " ";
      cnt += 3;
    }else{
      n-=1;
      cout << cnt << " ";
      cnt += 1;
    }
  }
  cout << 1 << '\n';
}