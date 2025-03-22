def verify[T](v: T | None) -> T:
  if v is None:
    raise TypeError()
  return v

