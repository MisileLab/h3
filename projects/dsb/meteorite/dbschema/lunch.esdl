module Lunch {
  type School {
    required name: str {constraint exclusive;}
    required school_code: int64 {constraint exclusive;}
    required ofcdc_code: str {constraint exclusive;}
  }
}
