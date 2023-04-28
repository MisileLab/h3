import * as readline from 'node:readline';
import * as process from 'node:process';

let rl = readline.createInterface({
    input: process.stdin,
    output: process.stdout
});

let me = {
    "Computer": "Lg Gram",
    "Programming Languages": ["Rust", "Python", "JS", "etc"],
    "OS": "Linux",
    "Bitcoin Price": 37909732.76,
    "AnA": true
}

function sum(a: Array<number>): number {
    let sum = 0;
    for (const element of a) {
        sum += element
    }
    return sum;
}

function sort(a: Array<number>): Array<number> {
    return a.sort((a, b) => a - b);
}

function center(a: Array<number>): number {
    let b = sort(a)
    return b[Math.floor(b.length / 2)]
}

function arrayer(a: Array<number>): [number, number] {
    return [sum(a) / a.length, center(a)]
}

// 1. avg, 2. center
console.log(arrayer(prompt("Enter a numbers: ")!.split(" ").map(Number)))

/*3.
boolean = (true, false)
null = 없는 값
undefined = 할당하지 않은 값
Number = 숫자
String = 문자열
Object = 객체
*/
