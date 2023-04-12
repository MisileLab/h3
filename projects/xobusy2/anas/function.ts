import { stdout } from "node:process";

function add(a: number, b: number) {
    return a + b;
}

console.log(add(1000, 1000));

function check(a: number) {
    if (a > 10) {
        console.log("more than 10");
    } else if (a == 10) {
        console.log("== 10");
    } else {
        console.log("less than 10");
    }
}

check(40);
check(30);
check(20);
check(10);

function printStars(n: number) {
    for (let i=1;i<=n;i++) {
        stdout.write("*");
    }
}

printStars(10);
