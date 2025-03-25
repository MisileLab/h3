import Control.Monad (when)

main :: IO ()
main = do
    let sol = [(1,1), (7,5)]
    let loop sol = do
            let (m_prev, x_prev) = sol !! (length sol - 2)
            let (m_last, x_last) = sol !! (length sol - 1)
            let m_next = 6 * m_last - m_prev
            let x_next = 6 * x_last - x_prev
            let sol' = sol ++ [(m_next, x_next)]
            let n = (m_next + 1) `div` 2
            let b = (x_next + 1) `div` 2
            when (n > 10^12) $ do
                putStrLn $ "n = " ++ show n
                putStrLn $ "b = " ++ show b
            when (n <= 10^12) $ loop sol'
    loop sol
