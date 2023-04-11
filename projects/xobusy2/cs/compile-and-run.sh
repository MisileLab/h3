clang -O2 -pipe -march=native $1.c -o $1.out
echo compiled
./$1.out
