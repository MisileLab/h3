-- GPU kernel example
-- Demonstrates GPU effect tracking

addOneGpu: i32 -> i32 !{pure, gpu, alloc none}
addOneGpu x = x + 1

processCpu: i32 -> i32 !{pure, cpu, alloc none}
processCpu x = x * 2

main: i32
main = processCpu 21
