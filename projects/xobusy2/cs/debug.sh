clang `python mfc.py c debug` $1.c -o $1.out
echo compiled
./$1.out
