def a(x:str,y:bool):
 z=sum((1if i in'aeiou'else 0)for i in x)
 return z if y else len(x)-z

for i in[input() for _ in range(int(input()))]:
 print(i,int(a(i,True)>a(i,False)),sep='\n')
