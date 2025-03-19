sol :: Int -> Int
sol n = if n `mod` 3 == 0 || n `mod` 5 == 0
  then n 
  else 0

main =
  print (show (sum (map sol [3..999])))
