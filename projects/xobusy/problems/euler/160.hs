num = 1000000000000
modulo = 10^5
countTrailingZeros n = sum [n `div` 5^i | i <- [1..], i^5 <= n]

sol :: Int -> Int
sol n
  | n == 0 = 1
  | otherwise = result2 `mod` modulo
  where
    trailingZeros = countTrailingZeros num
    removeFactors num count = go num count
      where
        go x count
          | x `mod` 5 == 0 = go (x `div` 5) (count + 1)
          | even x = go (x `div` 2) (count + 1)
          | otherwise = (x, count)
    process (acc, count2) n =
      (newAcc `mod` modulo, newCount2)
      where
        (newNum, newCount2) = removeFactors num count2
        newAcc = acc * newNum
    (result, count2) = foldl process (1, 0) [1..n]

    excessTwo = count2 - trailingZeros
    finalResult = if excessTwo > 0 then (result * (2^excessTwo)) `mod` modulo else result

    result2 = finalResult

main = print (sol num)

