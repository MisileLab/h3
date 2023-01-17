const moo = require('moo')
const fs = require("fs")
const proc = require("process");

function equal(a, b) { return a == b }
function notequal(a, b) { return a != b }
function plus(a, b) { return a + b }
function range(size, startAt = 0) {
    return [...Array(size).keys()].map(i => i + startAt);
}

const _filec = fs.readFileSync(proc.argv[2], {encoding:'utf8', flag:'r'})
const _filecbuf = _filec.split('\n')
var _text = ""

function _lexing(_saver) {
    let lexer = moo.compile({
        WS:      /[ ]+/,
        comment: /\/\/.*?$|\/\*.*?\*\//,
        number:  /0|[1-9][0-9]*/,
        string:  /"(?:\\["\\]|[^\n"\\])*"/,
        name: /[]+\n/,
        sep: ':',
        semi: ',',
        lparen:  '(',
        rparen:  ')',
        dot: '.',
        assignmentOp: "=",
        identifier: /[a-zA-Z_][a-zA-Z0-9_]*/,
        keyword: ['while', 'if', 'else'],
        NL:      { match: /\n/, lineBreaks: true },
    })
    lexer.reset(_saver)
    while (true) {
        const _lexernext = lexer.next()
        if (_lexernext == undefined) {
            break
        }
        if (_lexernext.text.startsWith('_')) {
            console.log('no startswith _ please')
            break
        }
        if (_lexernext.type == 'comment') {
            continue
        } else if (_lexernext.type == 'identifier') {
            if (_lexernext.value == 'function') {
                lexer.next()
                var _name = lexer.next()
                if (_name.type != 'identifier') {
                    console.log('no name in function')
                    break
                }
                _text += `function ${_name.value} (`
                if (lexer.next().type != 'lparen') {
                    console.log('no lparen in function')
                    break
                }
                while (true) {
                    var _tok = lexer.next()
                    if (_tok.type != 'identifier' && _tok.type != 'WS' && _tok.type != 'semi' && _tok.type != 'sep') {
                        break
                    }
                    _text += _tok.value
                }
                if (!_text.startsWith(')')) {
                    _text += ')'
                    lexer.next()
                }
                while (true) {
                    var _tok = lexer.next()
                    if (_tok.type != 'WS') {
                        break
                    }
                }
                _text += '{'
                const _min = lexer.save().line
                var _max;
                while (true) {
                    const _tok2 = lexer.next()
                    if (_tok2.text == 'end') {
                        _max = _tok2.line
                        break
                    }
                }
                var _buffer = []
                for (const _num of range(_max - _min, _min)) {
                    _buffer.push(_filecbuf[_num - 1])
                }
                _lexing(_buffer.toString())
                _text += '}'
            } else {
                _text += _lexernext.value
            }
        } else {
            _text += _lexernext.text
        }
    }
}
_lexing(_filec)
eval(_text)
