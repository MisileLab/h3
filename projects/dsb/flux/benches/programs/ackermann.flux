ack: i32 -> i32 -> i32
ack m n =
  if m == 0 then
    n + 1
  else
    if n == 0 then
      ack (m - 1) 1
    else
      ack (m - 1) (ack m (n - 1))

main: i32
main = ack 3 6
