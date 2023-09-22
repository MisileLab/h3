use std::{
    fs::read_to_string,
    env::args, collections::HashMap
};

#[derive(PartialEq, Clone, Debug)]
enum TokenType {
    None,
    TokenScope,
    TokenFunScopeEnd,
    TokenFunScopeStart,
    TokenIdentifier,
    TokenVarName,
    TokenVarOptionKey,
    TokenVarOptionValue,
    TokenVarOptionSep,
    TokenFunStart,
    TokenFunName,
    TokenFunArgKey,
    TokenFunArgValue,
    TokenFunCall,
    TokenFunCallArg
}

#[derive(Debug)]
enum Type {
    String, Int, Float, Bool
}

#[derive(Debug)]
enum Value {
    String(String),
    Int(isize),
    Float(f64),
    Bool(bool)
}

#[derive(Debug)]
enum LexType {
    Function(String, HashMap<String, Type>, Vec<LexType>),
    Variable(Type, Value),
    FunctionCall(String, Vec<Value>)
}

fn compare_token_type(tokens: &Vec<Token>, token_type: TokenType) -> bool {
    if tokens.len() == 0 {
        return false;
    }
    return tokens[tokens.len()-1].token_type == token_type
}

#[derive(Clone, Debug)]
struct Token {
    token_type: TokenType,
    value: String
}

fn tokenizer(filecontent: Vec<String>) -> Vec<Token> {
    let mut buffer = String::new();
    let mut tokens: Vec<Token> = Vec::new();
    let mut i: usize = 0;
    while filecontent.get(i).is_some() {
        let buffer_without = buffer.strip_prefix("   ").unwrap_or(&buffer).strip_prefix(" ").unwrap_or(&buffer).strip_suffix(" ").unwrap_or(&buffer).strip_suffix("   ").unwrap_or(&buffer);
        let buffer_without2 = buffer.strip_prefix("   ").unwrap_or(&buffer).strip_prefix(" ").unwrap_or(&buffer);
        let mut token_found = false;
        let mut token_semi_found = false;
        let mut ignored = false;
        let t = &filecontent[i];
        let mut token = Token{token_type: TokenType::None, value: ";".to_string()};
        println!("{buffer_without2}");
        if (tokens.len() == 0 || compare_token_type(&tokens, TokenType::None) || compare_token_type(&tokens, TokenType::TokenScope) || compare_token_type(&tokens, TokenType::TokenFunScopeStart)) && buffer_without2 == "@ " {
            token_found = true;
            token.token_type = TokenType::TokenIdentifier;
            token.value = buffer.clone();
        } else if compare_token_type(&tokens, TokenType::TokenIdentifier) && buffer.ends_with("[") {
            token_found = true;
            token.token_type = TokenType::TokenVarName;
            token.value = buffer[0..buffer.len()-1].to_string()
        } else if compare_token_type(&tokens, TokenType::TokenIdentifier) && buffer == "/" {
            token_found = true;
            token.token_type = TokenType::TokenFunStart;
            token.value = buffer.clone();
        } else if compare_token_type(&tokens, TokenType::TokenFunStart) && buffer.ends_with(" ") {
            token_found = true;
            token.token_type = TokenType::TokenFunName;
            token.value = buffer[0..buffer.len()-1].to_string()
        } else if compare_token_type(&tokens, TokenType::TokenFunName) && t == "#" {
            let args = buffer.split(" ").map(|x| x.to_string()).collect::<Vec<String>>();
            for j in args {
                if j != "" {
                    let arg2 = j.split("@").map(|x| x.to_string()).collect::<Vec<String>>();
                    tokens.push(Token {
                        token_type: TokenType::TokenFunArgKey,
                        value: arg2[0].clone()
                    });
                    tokens.push(Token {
                        token_type: TokenType::TokenFunArgValue,
                        value: arg2[1].clone()
                    });
                }
            }
            buffer = String::new();
            while &filecontent[i] != "#" {
                i += 1;
            }
            tokens.push(Token {
                token_type: TokenType::TokenFunScopeStart,
                value: "#".to_string()
            });
            ignored = true;
        }
        else if (compare_token_type(&tokens, TokenType::TokenVarName) || compare_token_type(&tokens, TokenType::TokenVarOptionSep)) && buffer.ends_with("=") {
            token_found = true;
            token.token_type = TokenType::TokenVarOptionKey;
            token.value = buffer[0..buffer.len()-1].to_string()
        } else if compare_token_type(&tokens, TokenType::TokenVarOptionKey) && (buffer.ends_with("]") || buffer.ends_with(",")) {
            token_found = true;
            token.token_type = TokenType::TokenVarOptionValue;
            token.value = buffer[0..buffer.len()-1].to_string();
            if buffer.ends_with(",") {
                tokens.push(token.clone());
                token.token_type = TokenType::TokenVarOptionSep;
                token.value = ",".to_string();
            }
        } else if compare_token_type(&tokens, TokenType::None) && buffer_without2.starts_with("/") && buffer_without2.ends_with(" ") {
            token_found = true;
            token.token_type = TokenType::TokenFunCall;
            token.value = buffer_without[1..buffer_without.len()-1].to_string();
        } else if compare_token_type(&tokens, TokenType::TokenFunCall) && t == ";" {
            let args = buffer.split(" ").map(|x| x.to_string()).collect::<Vec<String>>();
            for j in args {
                tokens.push(Token {
                    token_type: TokenType::TokenFunCallArg,
                    value: j
                });
            }
            buffer = String::new();
        } else if compare_token_type(&tokens, TokenType::None) && buffer == "#" {
            token_found = true;
            token.token_type = TokenType::TokenFunScopeEnd;
            token.value = buffer.clone();
        } else if !compare_token_type(&tokens, TokenType::None) && buffer_without.ends_with("#") {
            token_found = true;
            token.token_type = TokenType::TokenScope;
            token.value = buffer_without.to_string();
        }
        if !compare_token_type(&tokens, TokenType::None) && t == ";" {
            token_semi_found = true;
        }
        if token_found {
            tokens.push(token);
            buffer = String::new();
        }
        if t != "\n" && !ignored {
            buffer.push_str(t.as_str());
        }
        if token_semi_found {
            let atoken = Token{token_type: TokenType::None, value: ";".to_string()};
            tokens.push(atoken);
            buffer = String::new();
        }
        i += 1;
    }
    return tokens;
}

fn parse(tokens: &Vec<Token>, i: Option<usize>, end: Option<usize>) -> Vec<LexType> {
    let mut i = i.unwrap_or(0);
    let end = end.unwrap_or(tokens.len());
    let mut lexs: Vec<LexType> = Vec::new();
    while tokens.get(i).is_some() && i <= end {
        let t = &tokens[i];
        if t.token_type == TokenType::TokenFunStart {
            i += 2;
            if tokens[i-1].token_type == TokenType::TokenFunName {
                let mut args: HashMap<String, Type> = HashMap::new();
                let mut scope: Vec<LexType> = Vec::new();
                let name = tokens[i-1].value.clone();
                while tokens[i].token_type == TokenType::TokenFunArgKey && tokens[i+1].token_type == TokenType::TokenFunArgValue {
                    let typearg: Type;
                    if tokens[i+1].value == "int" {
                        typearg = Type::String;
                    } else if tokens[i+1].value == "string" {
                        typearg = Type::Int;
                    } else if tokens[i+1].value == "bool" {
                        typearg = Type::Bool;
                    } else if tokens[i+1].value == "float" {
                        typearg = Type::Float;
                    } else {
                        panic!("error when parse type")
                    }
                    args.insert(tokens[i].value.to_string(), typearg);
                    i += 2;
                }
                let i2 = i;
                while tokens[i].token_type != TokenType::TokenFunScopeEnd {
                    i += 1;
                }
                scope.extend(parse(&tokens, Some(i2), Some(i)));
                lexs.push(LexType::Function(name, args, scope));
            }
        } else if t.token_type == TokenType::TokenVarName {
            i += 4;
            let typearg: Type;
            let rvalue = &tokens[i+1].value;
            let value: Value;
            if tokens[i-2].value == "string" {
                typearg = Type::String;
                value = Value::String(rvalue.to_owned());
            } else if tokens[i-2].value == "int" {
                typearg = Type::Int;
                value = Value::Int(rvalue.parse().expect("error while parsing int"))
            } else if tokens[i-2].value == "bool" {
                typearg = Type::Bool;
                value = Value::Bool(if rvalue == "true" {true} else {false})
            } else if tokens[i-2].value == "float" {
                typearg = Type::Float;
                value = Value::Float(rvalue.parse().expect("error while parsing float"))
            } else {
                panic!("error when parse type")
            }
            lexs.push(LexType::Variable(typearg, value));
        } else if t.token_type == TokenType::TokenFunCall {
            let name = t.value.strip_prefix("   ").unwrap_or(&t.value);
            let name = name[1..name.len()].to_string();
            i += 1;
            let mut arguments: Vec<Value> = Vec::new();
            while tokens[i].token_type == TokenType::TokenFunCallArg {
                match &tokens[i].value.parse::<f64>() {
                    Ok(a) => {
                        arguments.push(Value::Float(a.to_owned()));
                        continue;
                    }, Err(_) => {}
                }
                match &tokens[i].value.parse::<isize>() {
                    Ok(a) => {
                        arguments.push(Value::Int(a.to_owned()));
                        continue;
                    }, Err(_) => {}
                }
                if &tokens[i].value == "true" {
                    arguments.push(Value::Bool(true))
                } else if &tokens[i].value == "false" {
                    arguments.push(Value::Bool(false))
                } else {
                    arguments.push(Value::String(tokens[i].value.clone()))
                }
                i += 1;
            }
            lexs.push(LexType::FunctionCall(name, arguments))
        }
        i += 1;
    }
    return lexs;
}

fn main() {
    let tmp = args().collect::<Vec<String>>();
    let filename: String = (&tmp[1]).to_owned();
    let filecontent: Vec<String> = read_to_string(filename).unwrap().chars().map(|c| c.to_string()).collect();
    let tokens = tokenizer(filecontent);
    for i in &tokens {
        println!("type: {:#?}, buffer: {:#?}", i.token_type, i.value);
    }
    let lexs = parse(&tokens, None, None);
    for i in &lexs {
        println!("lextype: {:#?}", i);
    }
}
