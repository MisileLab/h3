// Rust implementations of benchmark programs for comparison

pub fn fibonacci(n: i32) -> i32 {
    if n < 2 {
        n
    } else {
        fibonacci(n - 1) + fibonacci(n - 2)
    }
}

pub fn factorial(n: i32) -> i32 {
    if n <= 1 {
        1
    } else {
        n * factorial(n - 1)
    }
}

pub fn sum(n: i32) -> i32 {
    if n <= 0 {
        0
    } else {
        n + sum(n - 1)
    }
}

pub fn ackermann(m: i32, n: i32) -> i32 {
    if m == 0 {
        n + 1
    } else if n == 0 {
        ackermann(m - 1, 1)
    } else {
        ackermann(m - 1, ackermann(m, n - 1))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fibonacci() {
        assert_eq!(fibonacci(10), 55);
        assert_eq!(fibonacci(20), 6765);
    }

    #[test]
    fn test_factorial() {
        assert_eq!(factorial(5), 120);
        assert_eq!(factorial(10), 3628800);
    }

    #[test]
    fn test_sum() {
        assert_eq!(sum(10), 55);
        assert_eq!(sum(100), 5050);
    }

    #[test]
    fn test_ackermann() {
        assert_eq!(ackermann(3, 4), 125);
        assert_eq!(ackermann(3, 6), 509);
    }
}
