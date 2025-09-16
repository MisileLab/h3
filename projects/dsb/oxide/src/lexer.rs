// src/lexer.rs

use std::fmt;

#[derive(Debug, Clone, PartialEq)]
pub enum Token {
    // Keywords
    Data,      // data
    Where,     // where
    Case,      // case
    Of,        // of
    Cls,       // cls
    Inst,      // inst
    Use,       // use
    Module,    // module
    Foreign,   // foreign

    // Literals
    Nat(u64),
    Float(f64),
    String(String),

    // Identifiers
    Identifier(String),

    // Operators & Punctuation
    Colon,       // :
    Equals,      // =
    Pipe,        // |
    Arrow,       // ->
    At,          // @
    Hash,        // #
    Plus,        // +
    Minus,       // -
    Star,        // *
    Slash,       // /
    GreaterThan, // >
    LessThan,    // <

    // Delimiters
    LBrace, // {
    RBrace, // }
    LParen, // (
    RParen, // )

    // Special
    Eof, // End of File
}

impl fmt::Display for Token {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}", self)
    }
}

// A simple lexer function for now.
// It's not very efficient or robust, but it's a start.
pub fn lex(input: &str) -> Result<Vec<Token>, String> {
    let mut tokens = Vec::new();
    let mut chars = input.chars().peekable();

    while let Some(&c) = chars.peek() {
        match c {
            ' ' | '\r' | '\t' | '\n' => {
                // Skip whitespace
                chars.next();
            }
            '{' => {
                tokens.push(Token::LBrace);
                chars.next();
            }
            '}' => {
                tokens.push(Token::RBrace);
                chars.next();
            }
            '(' => {
                tokens.push(Token::LParen);
                chars.next();
            }
            ')' => {
                tokens.push(Token::RParen);
                chars.next();
            }
            ':' => {
                tokens.push(Token::Colon);
                chars.next();
            }
            '=' => {
                tokens.push(Token::Equals);
                chars.next();
            }
            '|' => {
                tokens.push(Token::Pipe);
                chars.next();
            }
            '@' => {
                tokens.push(Token::At);
                chars.next();
            }
            '#' => {
                tokens.push(Token::Hash);
                chars.next();
            }
            '+' => {
                tokens.push(Token::Plus);
                chars.next();
            }
            '*' => {
                tokens.push(Token::Star);
                chars.next();
            }
            '/' => {
                // Handle comments `//`
                chars.next(); // consume '/'
                if let Some('/') = chars.peek() {
                    while let Some(next_char) = chars.next() {
                        if next_char == '\n' {
                            break;
                        }
                    }
                } else {
                    tokens.push(Token::Slash);
                }
            }
            '-' => {
                // Handle arrow `->`
                chars.next(); // consume '-'
                if let Some('>') = chars.peek() {
                    chars.next(); // consume '>'
                    tokens.push(Token::Arrow);
                } else {
                    tokens.push(Token::Minus);
                }
            }
            c if c.is_ascii_digit() => {
                let mut num_str = String::new();
                while let Some(&d) = chars.peek() {
                    if d.is_ascii_digit() || d == '.' {
                        num_str.push(d);
                        chars.next();
                    } else {
                        break;
                    }
                }
                if num_str.contains('.') {
                    tokens.push(Token::Float(num_str.parse().map_err(|e| format!("Invalid float: {}", e))?));
                } else {
                    tokens.push(Token::Nat(num_str.parse().map_err(|e| format!("Invalid nat: {}", e))?));
                }
            }
            c if c.is_alphabetic() || c == '_' => {
                let mut ident = String::new();
                while let Some(&i) = chars.peek() {
                    if i.is_alphanumeric() || i == '_' {
                        ident.push(i);
                        chars.next();
                    } else {
                        break;
                    }
                }
                // Check for keywords
                match ident.as_str() {
                    "data" => tokens.push(Token::Data),
                    "where" => tokens.push(Token::Where),
                    "case" => tokens.push(Token::Case),
                    "of" => tokens.push(Token::Of),
                    "cls" => tokens.push(Token::Cls),
                    "inst" => tokens.push(Token::Inst),
                    "use" => tokens.push(Token::Use),
                    "module" => tokens.push(Token::Module),
                    "foreign" => tokens.push(Token::Foreign),
                    _ => tokens.push(Token::Identifier(ident)),
                }
            }
            _ => {
                return Err(format!("Unexpected character: {}", c));
            }
        }
    }

    tokens.push(Token::Eof);
    Ok(tokens)
}
