use std::fmt;

#[derive(Debug, Clone, PartialEq)]
pub enum Token {
    // Literals
    IntLit(i128),
    FloatLit(f64),
    StringLit(String),
    CharLit(char),
    BoolLit(bool),

    // Identifiers and keywords
    Ident(String),

    // Keywords
    Let,
    In,
    If,
    Then,
    Else,
    Match,
    With,
    Data,
    Type,
    Unsafe,

    // Effect keywords
    Pure,
    Io,
    State,
    Debug,
    Cpu,
    Gpu,
    Alloc,
    None_,
    Arena,
    Heap,
    Concurrent,
    Single,

    // Parallel primitives
    ParFor,
    ParMap,
    ParMapInplace,

    // Async primitives
    Async,
    Await,

    // Actor primitives
    Spawn,
    Send,
    Receive,

    // Memory primitives
    NewArena,
    AllocArray,

    // GPU primitives
    GpuKernel,
    CpuToGpu,
    GpuToCpu,

    // Debug primitives
    Log,
    Assert,

    // Operators
    Plus,
    Minus,
    Star,
    Slash,
    Percent,

    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,

    And,
    Or,
    Not,

    // Delimiters
    LParen,
    RParen,
    LBrace,
    RBrace,
    LBracket,
    RBracket,

    // Punctuation
    Arrow,      // ->
    FatArrow,   // =>
    Colon,
    Comma,
    Pipe,
    Bang,       // !
    Dot,
    Underscore,

    // Special
    Eof,
    Newline,
}

impl fmt::Display for Token {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Token::IntLit(n) => write!(f, "{}", n),
            Token::FloatLit(n) => write!(f, "{}", n),
            Token::StringLit(s) => write!(f, "\"{}\"", s),
            Token::CharLit(c) => write!(f, "'{}'", c),
            Token::BoolLit(b) => write!(f, "{}", b),
            Token::Ident(s) => write!(f, "{}", s),
            _ => write!(f, "{:?}", self),
        }
    }
}

pub struct Lexer {
    input: Vec<char>,
    position: usize,
    current: Option<char>,
}

impl Lexer {
    pub fn new(input: &str) -> Self {
        let chars: Vec<char> = input.chars().collect();
        let current = chars.get(0).copied();
        Lexer {
            input: chars,
            position: 0,
            current,
        }
    }

    fn advance(&mut self) {
        self.position += 1;
        self.current = self.input.get(self.position).copied();
    }

    fn peek(&self, offset: usize) -> Option<char> {
        self.input.get(self.position + offset).copied()
    }

    fn skip_whitespace(&mut self) {
        while let Some(ch) = self.current {
            if ch.is_whitespace() && ch != '\n' {
                self.advance();
            } else {
                break;
            }
        }
    }

    fn skip_comment(&mut self) {
        // Skip -- style comments
        if self.current == Some('-') && self.peek(1) == Some('-') {
            while self.current.is_some() && self.current != Some('\n') {
                self.advance();
            }
        }
    }

    fn read_number(&mut self) -> Token {
        let mut num_str = String::new();
        let mut is_float = false;

        while let Some(ch) = self.current {
            if ch.is_ascii_digit() {
                num_str.push(ch);
                self.advance();
            } else if ch == '.' && self.peek(1).map_or(false, |c| c.is_ascii_digit()) {
                is_float = true;
                num_str.push(ch);
                self.advance();
            } else {
                break;
            }
        }

        if is_float {
            Token::FloatLit(num_str.parse().unwrap())
        } else {
            Token::IntLit(num_str.parse().unwrap())
        }
    }

    fn read_string(&mut self) -> Token {
        self.advance(); // Skip opening "
        let mut s = String::new();

        while let Some(ch) = self.current {
            if ch == '"' {
                self.advance();
                break;
            } else if ch == '\\' {
                self.advance();
                match self.current {
                    Some('n') => s.push('\n'),
                    Some('t') => s.push('\t'),
                    Some('r') => s.push('\r'),
                    Some('\\') => s.push('\\'),
                    Some('"') => s.push('"'),
                    _ => s.push('\\'),
                }
                self.advance();
            } else {
                s.push(ch);
                self.advance();
            }
        }

        Token::StringLit(s)
    }

    fn read_char(&mut self) -> Token {
        self.advance(); // Skip opening '
        let ch = self.current.unwrap_or('\0');
        self.advance();

        if self.current == Some('\'') {
            self.advance();
        }

        Token::CharLit(ch)
    }

    fn read_ident(&mut self) -> Token {
        let mut ident = String::new();

        while let Some(ch) = self.current {
            if ch.is_alphanumeric() || ch == '_' {
                ident.push(ch);
                self.advance();
            } else {
                break;
            }
        }

        match ident.as_str() {
            "let" => Token::Let,
            "in" => Token::In,
            "if" => Token::If,
            "then" => Token::Then,
            "else" => Token::Else,
            "match" => Token::Match,
            "with" => Token::With,
            "data" => Token::Data,
            "type" => Token::Type,
            "unsafe" => Token::Unsafe,
            "true" => Token::BoolLit(true),
            "false" => Token::BoolLit(false),

            // Effect keywords
            "pure" => Token::Pure,
            "io" => Token::Io,
            "state" => Token::State,
            "debug" => Token::Debug,
            "cpu" => Token::Cpu,
            "gpu" => Token::Gpu,
            "alloc" => Token::Alloc,
            "none" => Token::None_,
            "arena" => Token::Arena,
            "heap" => Token::Heap,
            "concurrent" => Token::Concurrent,
            "single" => Token::Single,

            // Parallel primitives
            "par_for" => Token::ParFor,
            "par_map" => Token::ParMap,
            "par_map_inplace" => Token::ParMapInplace,

            // Async primitives
            "async" => Token::Async,
            "await" => Token::Await,

            // Actor primitives
            "spawn" => Token::Spawn,
            "send" => Token::Send,
            "receive" => Token::Receive,

            // Memory primitives
            "new_arena" => Token::NewArena,
            "alloc_array" => Token::AllocArray,

            // GPU primitives
            "gpu_kernel" => Token::GpuKernel,
            "cpu_to_gpu" => Token::CpuToGpu,
            "gpu_to_cpu" => Token::GpuToCpu,

            // Debug primitives
            "log" => Token::Log,
            "assert" => Token::Assert,

            _ => Token::Ident(ident),
        }
    }

    pub fn next_token(&mut self) -> Token {
        loop {
            self.skip_whitespace();

            if self.current == Some('-') && self.peek(1) == Some('-') {
                self.skip_comment();
                continue;
            }

            break;
        }

        match self.current {
            None => Token::Eof,
            Some('\n') => {
                self.advance();
                Token::Newline
            }
            Some('+') => {
                self.advance();
                Token::Plus
            }
            Some('*') => {
                self.advance();
                Token::Star
            }
            Some('/') => {
                self.advance();
                Token::Slash
            }
            Some('%') => {
                self.advance();
                Token::Percent
            }
            Some('(') => {
                self.advance();
                Token::LParen
            }
            Some(')') => {
                self.advance();
                Token::RParen
            }
            Some('{') => {
                self.advance();
                Token::LBrace
            }
            Some('}') => {
                self.advance();
                Token::RBrace
            }
            Some('[') => {
                self.advance();
                Token::LBracket
            }
            Some(']') => {
                self.advance();
                Token::RBracket
            }
            Some(':') => {
                self.advance();
                Token::Colon
            }
            Some(',') => {
                self.advance();
                Token::Comma
            }
            Some('|') => {
                self.advance();
                Token::Pipe
            }
            Some('.') => {
                self.advance();
                Token::Dot
            }
            Some('_') => {
                self.advance();
                Token::Underscore
            }
            Some('-') => {
                self.advance();
                if self.current == Some('>') {
                    self.advance();
                    Token::Arrow
                } else {
                    Token::Minus
                }
            }
            Some('=') => {
                self.advance();
                if self.current == Some('>') {
                    self.advance();
                    Token::FatArrow
                } else if self.current == Some('=') {
                    self.advance();
                    Token::Eq
                } else {
                    // Assignment not supported as token, use in parser
                    Token::Eq
                }
            }
            Some('!') => {
                self.advance();
                if self.current == Some('=') {
                    self.advance();
                    Token::Ne
                } else {
                    Token::Bang
                }
            }
            Some('<') => {
                self.advance();
                if self.current == Some('=') {
                    self.advance();
                    Token::Le
                } else {
                    Token::Lt
                }
            }
            Some('>') => {
                self.advance();
                if self.current == Some('=') {
                    self.advance();
                    Token::Ge
                } else {
                    Token::Gt
                }
            }
            Some('&') => {
                self.advance();
                if self.current == Some('&') {
                    self.advance();
                    Token::And
                } else {
                    Token::And
                }
            }
            Some('"') => self.read_string(),
            Some('\'') => self.read_char(),
            Some(ch) if ch.is_ascii_digit() => self.read_number(),
            Some(ch) if ch.is_alphabetic() || ch == '_' => self.read_ident(),
            Some(ch) => {
                self.advance();
                Token::Ident(ch.to_string())
            }
        }
    }

    pub fn tokenize(&mut self) -> Vec<Token> {
        let mut tokens = Vec::new();
        loop {
            let token = self.next_token();
            if token == Token::Eof {
                tokens.push(token);
                break;
            }
            // Keep newlines for multi-line function declarations
            tokens.push(token);
        }
        tokens
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_tokens() {
        let mut lexer = Lexer::new("let x = 42");
        let tokens = lexer.tokenize();
        assert_eq!(tokens[0], Token::Let);
        assert_eq!(tokens[1], Token::Ident("x".to_string()));
        assert_eq!(tokens[3], Token::IntLit(42));
    }

    #[test]
    fn test_effects() {
        let mut lexer = Lexer::new("!{pure, cpu, alloc none}");
        let tokens = lexer.tokenize();
        assert!(tokens.contains(&Token::Pure));
        assert!(tokens.contains(&Token::Cpu));
        assert!(tokens.contains(&Token::Alloc));
        assert!(tokens.contains(&Token::None_));
    }
}
