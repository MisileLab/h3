use std::{io::stdin as _stdin, iter::zip};

fn mean(numbers: &Vec<i32>) -> f32 {
  let sum: i32 = numbers.iter().sum();
  sum as f32 / numbers.len() as f32
}

fn median(numbers: &Vec<i32>) -> f32 {
  let mid = numbers.len() / 2;
  if numbers.len() % 2 == 0 {
    mean(&vec![numbers[mid - 1], numbers[mid]]) as f32
  } else {
    numbers[mid] as f32
  }
}

pub fn main() {
  let stdin = _stdin();
  let mut a = Vec::<Vec<i32>>::new();
  loop {
    let mut b = String::new();
    let mut d = Vec::<i32>::new();
    let c = stdin.read_line(&mut b);
    match c {
      Ok(_) => {
        if b == "0\n" || b == "" {
          break;
        }
        let mut e: Vec<String> = b.split_ascii_whitespace().map(|s| s.to_string()).collect();
        e.remove(0);
        for i in e {
          d.push(i.parse().unwrap());
        }
        a.push(d);
      },
      Err(_) => { break }
    }
  }
  for (i, i2) in zip(&a, 1..a.len()+1) {
    println!("Case {i2}: {:.1}", median(i));
  }
}
