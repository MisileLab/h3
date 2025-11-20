fact: i32 -> i32
fact n =
  if n <= 1 then
    1
  else
    n * fact (n - 1)

main: i32
main = fact 15
