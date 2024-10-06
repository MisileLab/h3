from pathlib import Path

class SlidingWindowedByte:
  def __init__(self, distance: int, length: int, value: int):
    self.distance = distance # 중복되는 문자와의 거리
    self.length = length # 중복되는 문자의 길이
    self.value = value # 문자의 값

  def to_tuple(self) -> tuple[int, int, int]:
    return (self.distance, self.length, self.value)

  def to_str_tuple(self) -> tuple[int, int, int]:
    return (self.distance, self.length, chr(self.value))

class ArithmeticCoding:
  # [0, 1]로 가정하여 인코딩
  def encode(self, data: list[SlidingWindowedByte], freq_table: dict): # 나중에 dict 세부 타입 작성 예정
    # 일정 비율로 나눈 뒤에 뒷 부분에서 작업 진행
    for char in data:
      self.high = freq_table[char][1]
      self.low = freq_table[char][0]
    # 마지막에 2로 나눔 (아직 이해하지 못함)
    return (self.low + self.high) / 2

#   def decode(self, encoded_value, freq_table, length):
#     decoded_data = []
#     for _ in range(length):
#       for char, (low_range, high_range) in freq_table.items():
#         if low_range <= encoded_value < high_range:
#           decoded_data.append(char)
#           char_range = high_range - low_range
#           encoded_value = (encoded_value - low_range) / char_range
#           break
#     return ''.join(decoded_data)

# def build_frequency_table(data):
#   freq_table = {}
#   total_chars = len(data)
  
#   for char in data:
#     if char in freq_table:
#       freq_table[char] += 1
#     else:
#       freq_table[char] = 1

#   # 빈도 테이블을 (low_range, high_range)의 범위로 변환
#   freq_table_ranges = {}
#   low_range = 0.0
#   for char, freq in sorted(freq_table.items()):
#     high_range = low_range + (freq / total_chars)
#     freq_table_ranges[char] = (low_range, high_range)
#     low_range = high_range
  
#   return freq_table_ranges

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

def compress_bytes(data: bytes) -> bytes:
  raise NotImplementedError()

def decompress_bytes(data: bytes) -> bytes:
  raise NotImplementedError()

def compress(file_path: str, output_path: str):
  Path(output_path).write_bytes(compress_bytes(Path(file_path).read_bytes()))

def decompress(file_path: str, output_path: str):
  Path(output_path).write_bytes(decompress_bytes(Path(file_path).read_bytes()))

if __name__ == '__main__':
  print('HellABHello', [i.to_str_tuple() for i in sliding_window(b'HellABHello')])
