-- Parallel computation example
-- Demonstrates concurrent effect tracking

processItem: i32 -> i32
processItem x = x * x

parallelProcess: i32 -> i32 !{pure, cpu, alloc none, concurrent}
parallelProcess n = processItem n

main: i32
main = parallelProcess 7
