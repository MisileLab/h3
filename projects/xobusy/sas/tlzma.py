class SlidingWindowedByte:
  def __init__(self, distance: int, length: int, value: bytes):
    if isinstance(value, int):
      raise ValueError("value must be bytes")
    self.distance = distance # 중복되는 문자와의 거리
    self.length = length # 중복되는 문자의 길이
    self.value = value # 문자의 값

  def to_tuple(self) -> tuple[int, int, bytes]:
    return (self.distance, self.length, self.value)

  def to_str_tuple(self) -> tuple[int, int, str]:
    return (self.distance, self.length, self.value.decode())

  def __repr__(self) -> tuple[int, int, str]:
    return self.to_str_tuple().__str__()

# (0, 1)
class ArithmeticCoding:
  def encode(
    self,
    data: list[SlidingWindowedByte],
    freq_table: dict[
      tuple[int, int, bytes],
      tuple[float, float]
  ]) -> float:
    # 일정 비율로 나눈 뒤에 뒷 부분에서 작업 진행
    high, low = 0, 0
    for window in data:
      high = freq_table[window.to_tuple()][1]
      low = freq_table[window.to_tuple()][0]
    # 압축은 결과로 하나의 값을 줘야 하기 때문에 평균값을 리턴
    return (low + high) / 2

  def decode(
    self,
    data: float,
    freq_table: dict[
      tuple[int, int, bytes],
      tuple[float, float]
    ],
    length: int
  ) -> list[SlidingWindowedByte]:
    value = []
    for _ in range(length):
      for window, (low, high) in freq_table.items():
        # 만약 범위 사이에 값이 있다면 값을 얻은 후에 다시 시도
        if low <= data < high:
          value.append(window)
          # 수식 최적화 가능할 것 같음
          data = (data - low) / (high - low)
          break
    return [SlidingWindowedByte(*i) for i in value]

# 확률 테이블 생성
def build_frequency_table(data: list[SlidingWindowedByte]):
  freqs: dict[tuple[int, int, bytes], float] = {}
  length = len(data)

  # 확률(빈도 수) 계산
  for i in data:
    freqs[i.to_tuple()] = freqs[i.to_tuple()] + 1 if i in freqs else 1
  print(freqs)

  freq_ranges = {}
  # (0, 1)
  low = 0.0
  # 범위 생성
  for char, freq in freqs.items():
    high = low + freq / length
    freq_ranges[char] = (low, high)
    low = high
  
  return freq_ranges

def find_distance(data: bytes, compare: bytes, org_index: int) -> int:
  queue = []
  first_char = -1
  for i in range(len(data)):
    if data[i] == compare[len(queue)]:
      if first_char == -1:
        first_char = i
      queue.append(data[i])
    else:
      first_char = -1
      queue.clear()
    if bytes(queue) == compare:
      print(org_index, queue)
      return org_index-len(queue)*2 if len(queue) == 1 else (org_index-len(queue)*2+1)-2-len(queue)
  raise ValueError("unreachable")

# https://dalinaum.github.io/algorithm/2020/12/14/zip-compression.html
def sliding_window(data: bytes) -> list[SlidingWindowedByte]:
  result = [SlidingWindowedByte(0, 0, data[0].to_bytes())]
  duplicated = 0
  ld = len(data)
  i = 0
  while i < ld-1:
    value = data[i+1]
    if value not in data[:i+1]:
      # [i-duplicated:i]로 하면 (i-duplicated+1: i]가 되기 때문에 한 글자 땡겨줘야 함
      result.append(SlidingWindowedByte(
        0 if duplicated == 0 else find_distance(data[:i],data[i-duplicated+1:i+1],i),
        duplicated,
        b'' if duplicated != 0 else value.to_bytes()
      ))
      if duplicated != 0:
        result.append(SlidingWindowedByte(
          0, 0,
          value.to_bytes()
        ))
      duplicated = 0
      i += 1
      continue
    duplicated += 1
    i += 1
  return result

def decode_sliding_window(data: list[SlidingWindowedByte]) -> bytes:
  result = b""
  for i, j in enumerate(data):
    if j.distance == 0:
      result += j.value
    else:
      print(result, i-j.distance-j.length+1, i-j.distance+1)
      start = len(result)-j.distance-j.length if j.distance != j.length else len(result)-j.distance
      print(result[start].to_bytes())
      i = 0
      while i < j.length:
        result += result[start+i].to_bytes()
        i += 1
  return result

def compress(file_path: str, output_path: str):
  raise NotImplementedError

def decompress(file_path: str, output_path: str):
  raise NotImplementedError

if __name__ == '__main__':
  sld = sliding_window(b'HelloABCimABCD')
  print(sld)
  print(decode_sliding_window(sld))
  frq = build_frequency_table(sld)
  enc = ArithmeticCoding().encode(sld, frq)
  dec = ArithmeticCoding().decode(enc, frq, len(sld))
  print(enc, decode_sliding_window(dec))
