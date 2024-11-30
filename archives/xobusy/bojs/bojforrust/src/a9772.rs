use std::io::stdin as _stdin;

pub fn main() {
  let stdin = _stdin();
  let mut a = Vec::<Vec<f32>>::new();
  loop {
    let mut b = String::new();
    let mut d = Vec::<f32>::new();
    let c = stdin.read_line(&mut b);
    match c {
      Ok(_) => {
        let e: Vec<String> = b.split_ascii_whitespace().map(|s| s.to_string()).collect();
        for i in e {
          d.push(i.parse().unwrap());
        }
        a.push(d);
        if b == "0 0\n" || b == "" {
          break;
        }
      },
      Err(_) => { break }
    }
  }
  for i in a {
    let (b, c) = (i[0], i[1]);
    if b == 0. || c == 0. {
      println!("AXIS");
    } else if b > 0. {
      if c > 0. {
        println!("Q1");
      } else {
        println!("Q4");
      }
    } else if c > 0. {
      println!("Q2");
    } else {
      println!("Q3");
    }
  }
}
