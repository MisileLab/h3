pub mod ast;
pub mod lexer;
pub mod parser;
pub mod types;
pub mod effect;
pub mod typecheck;
pub mod ir;
pub mod codegen;
pub mod backend_x86;
pub mod backend_wasm;
pub mod runtime;
pub mod actor;

use anyhow::Result;
use parser::Parser;
use typecheck::TypeChecker;
use codegen::CodeGenerator;
use backend_x86::X86Backend;
use backend_wasm::WasmBackend;

pub use ast::Module;
pub use types::TypeEnv;
pub use ir::IRModule;

/// Main compiler interface
pub struct FluxCompiler {
    type_checker: TypeChecker,
    code_generator: CodeGenerator,
}

impl FluxCompiler {
    pub fn new() -> Self {
        FluxCompiler {
            type_checker: TypeChecker::new(),
            code_generator: CodeGenerator::new(),
        }
    }

    /// Parse Flux source code into an AST
    pub fn parse(&self, source: &str) -> Result<Module> {
        let mut parser = Parser::new(source);
        parser.parse_module().map_err(|e| anyhow::anyhow!(e.message))
    }

    /// Type-check a Flux module
    pub fn check(&mut self, module: &Module) -> Result<TypeEnv> {
        let env = self
            .type_checker
            .check_module(module)
            .map_err(|e| anyhow::anyhow!(e.message))?;
        self.code_generator
            .set_literal_types(self.type_checker.literal_types().clone());
        Ok(env)
    }

    /// Generate IR from a checked module
    pub fn generate_ir(&mut self, module: &Module) -> Result<IRModule> {
        self.code_generator.generate(module)
            .map_err(|e| anyhow::anyhow!(e))
    }

    /// Compile to x86-64
    pub fn compile_x86(&mut self, ir: &IRModule) -> Result<X86Backend> {
        let mut backend = X86Backend::new()?;
        backend.compile(ir)?;
        Ok(backend)
    }

    /// Compile to WebAssembly
    pub fn compile_wasm(&mut self, ir: &IRModule) -> Result<Vec<u8>> {
        let mut backend = WasmBackend::new();
        backend.compile(ir)
    }

    /// Full compilation pipeline: source -> type check -> IR -> x86
    pub fn compile_source_x86(&mut self, source: &str) -> Result<X86Backend> {
        let module = self.parse(source)?;
        self.check(&module)?;
        let ir = self.generate_ir(&module)?;
        self.compile_x86(&ir)
    }

    /// Full compilation pipeline: source -> type check -> IR -> wasm
    pub fn compile_source_wasm(&mut self, source: &str) -> Result<Vec<u8>> {
        let module = self.parse(source)?;
        self.check(&module)?;
        let ir = self.generate_ir(&module)?;
        self.compile_wasm(&ir)
    }

    /// Just parse and type-check (for fluxc check command)
    pub fn check_source(&mut self, source: &str) -> Result<()> {
        let module = self.parse(source)?;
        let _ = self.check(&module)?;
        Ok(())
    }
}

impl Default for FluxCompiler {
    fn default() -> Self {
        Self::new()
    }
}

/// Compile and execute Flux code (for quick testing)
pub fn eval(source: &str) -> Result<i32> {
    let mut compiler = FluxCompiler::new();
    let backend = compiler.compile_source_x86(source)?;
    backend.execute_main()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_simple() {
        let compiler = FluxCompiler::new();
        let source = "inc: i32 -> i32\ninc x = x + 1";
        let result = compiler.parse(source);
        assert!(result.is_ok());
    }

    #[test]
    fn test_type_check_simple() {
        let mut compiler = FluxCompiler::new();
        let source = "inc: i32 -> i32\ninc x = x + 1";
        let result = compiler.check_source(source);
        assert!(result.is_ok());
    }

    #[test]
    fn test_effect_violation() {
        let mut compiler = FluxCompiler::new();
        let source = r#"
foo: i32 -> i32 !{pure, cpu, alloc heap}
foo x = x + 1

bar: i32 -> i32 !{pure, cpu, alloc none}
bar x = foo x
"#;
        let result = compiler.check_source(source);
        assert!(result.is_err());
    }

    #[test]
    fn test_eval_simple() {
        let source = r#"
main: i32
main = 42
"#;
        let result = eval(source);
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), 42);
    }

    #[test]
    fn test_eval_arithmetic() {
        let source = r#"
main: i32
main = 1 + 2 + 3
"#;
        let result = eval(source);
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), 6);
    }
}
