pub enum Statement {
    Block(Block),
    VariableDeclaration(VariableDeclaration),
    FunctionDeclaration(FunctionDeclaration),
}

pub struct Block {
    pub statements: Vec<Statement>,
}

pub struct VariableDeclaration {
    pub name: String,
    pub value: Value,
}

pub struct MutableVariableDeclaration {
    pub name: String,
    pub value: Value,
}

pub struct MutableValueVariableDeclaration {
    pub name: String,
    pub value: Value,
}

pub struct FunctionDeclaration {
    pub name: String,
    pub params: Vec<FunctionParam>,
    pub ret_type: Type,
    pub body: Vec<Statement>,
}

pub struct FunctionParam {
    pub name: String,
    pub t: Type,
}

pub enum Type {
    Custom(String),
    U8,
    U16,
    U32,
    U64,
    U128,
    I8,
    I16,
    I32,
    I64,
    I128,
    F32,
    F64,
    String,
}

pub enum Value {
    Integer(i64),
    Float(f64),
    String(String),
}
