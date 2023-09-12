use lalrpop_util::lalrpop_mod;
lalrpop_mod!(pub exsl); // generated parser

fn main() {
    let parser = exsl::StatementParser::new();
    let code = r#"
        a {
            b()
            c()
        }
        a = 0
        mut a = 0
        var a = 0
        func a(b: u64, c: u64, d: u64) -> u64 {
            ret b+c+d
        }
    "#;
    match parser.parse(code) {
        Ok(ast) => {
            // Do something with the AST...
        }
        Err(err) => eprintln!("Error parsing code: {:?}", err),
    }
}
