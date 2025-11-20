sum: i32 -> i32
sum n =
  if n <= 0 then
    0
  else
    n + sum (n - 1)

main: i32
main = sum 1000
