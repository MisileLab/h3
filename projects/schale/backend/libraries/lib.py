class nullError(Exception):
  pass

def nullVerify[T](v: T | None) -> T:
  if v is None:
    raise nullError()
  return v

