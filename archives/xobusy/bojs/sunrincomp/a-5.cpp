#include <iostream>

using namespace std;

int n,m;
int use;
int arr[2010][2010];
int main(){
  cin >> n >> m;
  for(int i=1;i<=1000;i++){
    for(int j=1;j<=1000;j++){
      if(i%2==1 && j%2 == 1){
        arr[i][j] = 1;
      }
      if(i%2==0 && j%2 == 1){
        arr[i][j] = 3;
      }
      if(i%2==1 && j%2==0){
        arr[i][j] = 2;
      }
      if(i%2==0 && j%2==0){
        arr[i][j] = 4;
      }
    }
  }if(n==1 && m==1){
    cout << 1 << '\n';
    cout << 1 << '\n';
  }
  else if(n==1 || m==1){
    if(n== 1){
    cout << 2 << '\n';
    for(int i=1;i<=n;i++){
      for(int j=1;j<=m;j++){
        cout << arr[i][j] << " ";
      }
      cout << '\n';
    }
    }else{
      cout << 2 << '\n';
      for(int i=1;i<=n;i++){
      for(int j=1;j<=m;j++){
        if(arr[i][j] == 3){
          cout << 2 << " ";
        }else{
          cout << arr[i][j] << " ";   
        }
      }
      cout << '\n';
    }
    }
  }else{
     cout << 4 << '\n';
     for(int i=1;i<=n;i++){
      for(int j=1;j<=m;j++){
        cout << arr[i][j] << " ";
      }
      cout << '\n';
    }
  }
}