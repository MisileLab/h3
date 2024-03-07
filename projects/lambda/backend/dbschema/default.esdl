module default {
  type User {
    constraint exclusive on (.name);
    required name: str;
    required pnumber: str;
    required me: str;
    required why: str;
    required time: float64;
    optional portfolio: str;    
  }
}
