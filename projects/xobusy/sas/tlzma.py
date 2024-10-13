from pathlib import Path

class SlidingWindowedByte:
  def __init__(self, distance: int, length: int, value: int):
    self.distance = distance # 중복되는 문자와의 거리
    self.length = length # 중복되는 문자의 길이
    self.value = value # 문자의 값

  def to_tuple(self) -> tuple[int, int, int]:
    return (self.distance, self.length, self.value)

  def to_str_tuple(self) -> tuple[int, int, str]:
    return (self.distance, self.length, chr(self.value))

# (0, 1)
class ArithmeticCoding:
  def encode(
    self,
    data: list[SlidingWindowedByte],
    freq_table: dict[
      SlidingWindowedByte,
      tuple[float, float]
  ]) -> float:
    # 일정 비율로 나눈 뒤에 뒷 부분에서 작업 진행
    high, low = 0, 0
    for window in data:
      high = freq_table[window][1]
      low = freq_table[window][0]
    # 압축은 결과로 하나의 값을 줘야 하기 때문에 평균값을 리턴
    return (low + high) / 2

  def decode(
    self,
    data: float,
    freq_table: dict[
      SlidingWindowedByte,
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
    return value

# 확률 테이블 생성
def build_frequency_table(data: list[SlidingWindowedByte]):
  freqs: dict[SlidingWindowedByte, float] = {}
  length = len(data)

  # 확률(빈도 수) 계산
  for char in data:
    freqs[char] = freqs[char] + 1 if char in freqs else 1
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

def find_distance(data: bytes, compare: bytes) -> int:
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
    if len(queue) == len(compare):
      return i-first_char+1
  raise ValueError("unreachable")

# https://dalinaum.github.io/algorithm/2020/12/14/zip-compression.html
def sliding_window(data: bytes) -> list[SlidingWindowedByte]:
  result = [SlidingWindowedByte(0, 0, data[0])]
  duplicated = 0
  ld = len(data)
  i = 0
  while i < ld-1:
    value = data[i+1]
    print(data[:i+1], chr(value))
    if value not in data[:i+1]:
      # [i-duplicated:i]로 하면 (i-duplicated+1: i]가 되기 때문에 한 글자 땡겨줘야 함
      result.append(SlidingWindowedByte(0 if duplicated == 0 else find_distance(data[:i], data[i-duplicated+1:i+1]), duplicated, value))
      duplicated = 0
      i += 1
      continue
    duplicated += 1
    i += 1
  return result

def decode_sliding_window(data: list[SlidingWindowedByte]) -> bytes:
  raise NotImplementedError

def compress(file_path: str, output_path: str):
  raise NotImplementedError

def decompress(file_path: str, output_path: str):
  raise NotImplementedError

if __name__ == '__main__':
  print('HellABHello', [i.to_str_tuple() for i in sliding_window(b'HellABHello')])
  sld = sliding_window(b'HellABHello')
  frq = build_frequency_table(sld)
  print(ArithmeticCoding().encode(sld, frq))
