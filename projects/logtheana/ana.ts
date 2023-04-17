// 1. Print just AnA
console.log("AnA")

// 2. Qwerty -> NaN -> AnA
console.log(String(Number("Qwerty")).replaceAll('N', '2').replaceAll('a', '1').replaceAll('2', 'A').replaceAll('1', 'n'))

// 3. Promisely AnA
Promise.any([
    Promise.reject(new SyntaxError("This is Error")),
]).catch(
    e => {
        console.log(e.name[0] + String(e.errors[0])[2] + e.name[0])
    }
)

// 4. Ana with array
console.log(["A", "n", "A"].join(""))

// 5. Generate AnA
function* ana(): Generator<String> {
    yield "A"
    yield "n"
    yield "A"
}

let anagen = ana()
console.log(anagen.next().value + anagen.next().value + anagen.next().value)

// 6. 01000001 01101110 01000001 = AnA
