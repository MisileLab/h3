-- Example demonstrating zero-allocation effect checking
-- This program should FAIL type checking because bar calls foo
-- but bar has alloc none while foo has alloc heap

foo: i32 -> i32 !{pure, cpu, alloc heap}
foo x = x + 1

bar: i32 -> i32 !{pure, cpu, alloc none}
bar x = foo x

main: i32
main = bar 42
