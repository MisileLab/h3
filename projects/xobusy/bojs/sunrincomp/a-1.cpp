#include <ios>
#include <iostream>
#include <ratio>
#include <vector>
#include <algorithm>
#include <math.h>

using namespace std;
int n,k,g;
int cnt = 0;
int arr[100010];
int main(){
  cin >> n >> k; 
  for(int i=1;i<=n;i++){
    cin >> arr[i];
  }
  sort(arr+1,arr+n+1,[&](auto a,auto b){
    return a < b;
  });
  for(int i=2;i<=n;i++){
    if(arr[1]+arr[i] <= k){
    g = max(g,i);
    }
  } 
  int i=1;
  int j=g;
  while(i < j){
    if(arr[i]+arr[j]<=k){
      i++;
      j--;
      cnt++;
    }else{
      j--;
    }
  }
  cout << cnt;
}