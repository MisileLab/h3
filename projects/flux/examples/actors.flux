-- Actor-based concurrency example
-- Demonstrates message-passing actor model

actorHandler: i32 -> i32 !{io, cpu, alloc heap, concurrent}
actorHandler msg = msg + 1

processMessage: i32 -> i32 !{io, cpu, alloc heap, concurrent}
processMessage x = actorHandler x

main: i32
main = processMessage 99
