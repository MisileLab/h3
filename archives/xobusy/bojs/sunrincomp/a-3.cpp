#include <iostream>
#include <vector>
#include <algorithm>

using namespace std;
using ll = long long;

ll n,k;
ll arr[1000010];
ll mmx;
ll sum;

bool go(ll mid){
  sum = 0;
  for(int i=0;i<n;i++){
    if(arr[i]-mid >= 0){
      sum+= arr[i]-mid;
    }
  }
  if(sum<=k){
    return 1;
  }else{
    return 0;
  }
}


int main(){
  cin >> n >> k;
  for(int i=0;i<n;i++){
    cin >> arr[i];
    mmx = max(arr[i],mmx);
  }
  ll l = 0,r = mmx+1; 
  while(l < r){
    ll mid = (l+r)/2;
    if(go(mid)){
      r = mid;
    }else{
      l = mid+1;
    }
  }
  cout<< l;
}