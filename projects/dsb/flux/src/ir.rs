// Intermediate Representation for Flux
// This is a lowered form suitable for both x86-64 and WebAssembly backends

#[derive(Debug, Clone)]
pub struct IRModule {
    pub functions: Vec<IRFunction>,
    pub globals: Vec<IRGlobal>,
}

#[derive(Debug, Clone)]
pub struct IRFunction {
    pub name: String,
    pub params: Vec<IRParam>,
    pub return_type: IRType,
    pub body: Vec<IRInst>,
    pub is_export: bool,
}

#[derive(Debug, Clone)]
pub struct IRParam {
    pub name: String,
    pub ty: IRType,
}

#[derive(Debug, Clone)]
pub struct IRGlobal {
    pub name: String,
    pub ty: IRType,
    pub init: Option<IRValue>,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum IRType {
    I32,
    I64,
    F32,
    F64,
    Bool,
    Ptr,
    Void,
    Struct(Vec<IRType>),
}

#[derive(Debug, Clone)]
pub enum IRInst {
    // Basic operations
    Assign(String, IRValue),
    BinOp(String, BinOp, IRValue, IRValue),
    UnOp(String, UnOp, IRValue),

    // Memory operations
    Load(String, IRValue),
    Store(IRValue, IRValue),
    Alloca(String, IRType, usize),

    // Control flow
    Label(String),
    Jump(String),
    CondJump(IRValue, String, String),
    Return(Option<IRValue>),

    // Function calls
    Call(Option<String>, String, Vec<IRValue>),

    // Phi node for SSA (simplified)
    Phi(String, Vec<(String, IRValue)>),
}

#[derive(Debug, Clone)]
pub enum IRValue {
    Var(String),
    I32(i32),
    I64(i64),
    F32(f32),
    F64(f64),
    Bool(bool),
    Null,
    GlobalRef(String),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BinOp {
    Add,
    Sub,
    Mul,
    Div,
    Mod,
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
    And,
    Or,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UnOp {
    Neg,
    Not,
}

impl IRModule {
    pub fn new() -> Self {
        IRModule {
            functions: Vec::new(),
            globals: Vec::new(),
        }
    }

    pub fn add_function(&mut self, func: IRFunction) {
        self.functions.push(func);
    }

    pub fn add_global(&mut self, global: IRGlobal) {
        self.globals.push(global);
    }

    pub fn find_function(&self, name: &str) -> Option<&IRFunction> {
        self.functions.iter().find(|f| f.name == name)
    }
}

impl Default for IRModule {
    fn default() -> Self {
        Self::new()
    }
}

impl IRFunction {
    pub fn new(name: String, params: Vec<IRParam>, return_type: IRType) -> Self {
        IRFunction {
            name,
            params,
            return_type,
            body: Vec::new(),
            is_export: false,
        }
    }

    pub fn add_inst(&mut self, inst: IRInst) {
        self.body.push(inst);
    }

    pub fn make_export(mut self) -> Self {
        self.is_export = true;
        self
    }
}

impl IRType {
    pub fn size_bytes(&self) -> usize {
        match self {
            IRType::I32 => 4,
            IRType::I64 => 8,
            IRType::F32 => 4,
            IRType::F64 => 8,
            IRType::Bool => 1,
            IRType::Ptr => 8, // Assuming 64-bit
            IRType::Void => 0,
            IRType::Struct(fields) => fields.iter().map(|f| f.size_bytes()).sum(),
        }
    }
}

// Builder for creating IR in a more ergonomic way
pub struct IRBuilder {
    current_function: Option<IRFunction>,
    temp_counter: usize,
    label_counter: usize,
}

impl IRBuilder {
    pub fn new() -> Self {
        IRBuilder {
            current_function: None,
            temp_counter: 0,
            label_counter: 0,
        }
    }

    pub fn start_function(&mut self, name: String, params: Vec<IRParam>, return_type: IRType) {
        self.current_function = Some(IRFunction::new(name, params, return_type));
        self.temp_counter = 0;
        self.label_counter = 0;
    }

    pub fn finish_function(&mut self) -> Option<IRFunction> {
        self.current_function.take()
    }

    pub fn fresh_temp(&mut self) -> String {
        let temp = format!("t{}", self.temp_counter);
        self.temp_counter += 1;
        temp
    }

    pub fn fresh_label(&mut self) -> String {
        let label = format!("L{}", self.label_counter);
        self.label_counter += 1;
        label
    }

    pub fn add_inst(&mut self, inst: IRInst) {
        if let Some(func) = &mut self.current_function {
            func.add_inst(inst);
        }
    }

    pub fn assign(&mut self, dest: String, value: IRValue) {
        self.add_inst(IRInst::Assign(dest, value));
    }

    pub fn binop(&mut self, dest: String, op: BinOp, left: IRValue, right: IRValue) {
        self.add_inst(IRInst::BinOp(dest, op, left, right));
    }

    pub fn unop(&mut self, dest: String, op: UnOp, operand: IRValue) {
        self.add_inst(IRInst::UnOp(dest, op, operand));
    }

    pub fn call(&mut self, dest: Option<String>, func: String, args: Vec<IRValue>) {
        self.add_inst(IRInst::Call(dest, func, args));
    }

    pub fn return_value(&mut self, value: Option<IRValue>) {
        self.add_inst(IRInst::Return(value));
    }

    pub fn label(&mut self, name: String) {
        self.add_inst(IRInst::Label(name));
    }

    pub fn jump(&mut self, target: String) {
        self.add_inst(IRInst::Jump(target));
    }

    pub fn cond_jump(&mut self, cond: IRValue, true_label: String, false_label: String) {
        self.add_inst(IRInst::CondJump(cond, true_label, false_label));
    }
}

impl Default for IRBuilder {
    fn default() -> Self {
        Self::new()
    }
}
