// 소유권 기반 메모리 관리
process : @List Nat -> List Nat = {
    process @xs = map (|x| x * 2) (filter (|x| x > 0) xs)
}

// 명시적 소유권 이동  
transfer : List a -> List a = {
    transfer list = compute (#list)  // # = 소유권 이동
}

// GPU 커널 정의
 @gpu @blockSize(256)
vectorAdd : Global (Array Float) -> Global (Array Float) -> Global (Array Float) = {
    vectorAdd xs ys = zipWith (+) xs ys
}
