clang -O2 -pipe $1.c -o $1.out
echo compiled
./$1.out
