fib :: Int -> Int
fib 0 = 0
fib 1 = 1
fib n = fib (n-1) + fib (n-2)
arr = takeWhile (< 4000000) [fib x | x <- [2..]]
res = [x | x <- arr, even x, x < 4000000]

main = print (show (sum res))
