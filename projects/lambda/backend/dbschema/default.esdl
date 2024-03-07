module default {
  type User {
    required name: str;
    required pnumber: str;
    required me: str;
    required why: str;
    required time: float64;
    optional portfolio: str;    
  }
}
