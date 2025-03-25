import Data.Maybe
import Data.List

sol :: Integer -> (Integer, Integer)
sol 1 = (1, 1)
sol 2 = (7, 5)
sol n = (n_next, b_next)
  where
    (m_prev, x_prev) = sol (n - 2)
    (m_last, x_last) = sol (n - 1)
    m_next = 6 * m_last - m_prev
    x_next = 6 * x_last - x_prev
    n_next = (m_next + 1) `div` 2
    b_next = (x_next + 1) `div` 2

res = [sol n | n <- [1..]]

main :: IO ()
main = do
  let answer = find ((> 10^12) . fst) $ map sol [1..]
  print $ fst <$> answer

