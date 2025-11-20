-- Basic Flux program demonstrating simple functions

inc: i32 -> i32
inc x = x + 1

add: i32 -> i32 -> i32
add x y = x + y

abs: i32 -> i32
abs x = if x < 0 then -x else x

main: i32
main =
  let x = 5 in
  let y = inc x in
  add y 10
