-- Async/await example
-- Demonstrates Task-based concurrency

compute: i32 -> i32 !{pure, cpu, alloc none}
compute x = x + 10

asyncCompute: i32 -> i32 !{io, cpu, alloc heap, concurrent}
asyncCompute x = compute x

main: i32
main = asyncCompute 32
