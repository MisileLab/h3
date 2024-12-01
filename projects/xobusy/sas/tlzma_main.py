from collections import Counter
from copy import deepcopy
import struct

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

  def __repr__(self) -> str:
    return self.to_str_tuple().__str__()

def convert_to_bytes(data: list[SlidingWindowedByte]):
  result = bytearray()
  for item in data:
    item = item.to_tuple()
    # Pack the integers (2 ints) and the string (as bytes) into the result
    result.extend(struct.pack('ii', item[0], item[1]))  # Pack two integers
    result.extend(item[2])  # Convert string to bytes
    result.append(0)  # Null byte separator (optional, for clarity)
  return bytes(result)

def build_frequency_table(sliding: list[SlidingWindowedByte]) -> dict[int, int]:
  c = Counter(convert_to_bytes(sliding))
  return dict(c)

def encode_range(frequencies: dict[int, int], values: bytes) -> int:
  total = sum(frequencies.values())
  # Build cumulative frequency table
  cum_freq = {}
  cum = 0
  for k in sorted(frequencies.keys()):
    cum_freq[k] = cum
    cum += frequencies[k]
  # Initialize low and high range values
  low = 0
  high = (1 << 32) - 1  # Using 32-bit range
  for i in values:
    range_ = high - low + 1
    high = low + (range_ * (cum_freq[i] + frequencies[i]) // total) - 1
    low = low + (range_ * cum_freq[i] // total)
  return low

def decode_range(frequencies: dict[int, int], code: int, length: int) -> bytes:
  total = sum(frequencies.values())
  symbols = sorted(frequencies.keys())
  cum_freq = {}
  cum = 0
  for symbol in symbols:
    cum_freq[symbol] = cum
    cum += frequencies[symbol]

  low = 0
  high = (1 << 32) - 1
  decoded = bytearray()

  for _ in range(length):
    range_ = high - low + 1
    value = ((code - low + 1) * total - 1) // range_

    for symbol in symbols:
      sym_low = cum_freq[symbol]
      sym_high = sym_low + frequencies[symbol]
      if sym_low <= value < sym_high:
        decoded.append(symbol)
        high = low + (range_ * sym_high // total) - 1
        low = low + (range_ * sym_low // total)
        break

  return bytes(decoded)

# https://dalinaum.github.io/algorithm/2020/12/14/zip-compression.html
def sliding_window(data: bytes) -> list[SlidingWindowedByte]:
  result = []
  i = 0

  while i < len(data):
    # 처음은 무조건 압축할 수 없기에 데이터 추가 후 넘김
    if i == 0:
      result.append(SlidingWindowedByte(0, 1, data[i:i+1]))
      i += 1
      continue

    max_length = 0
    distance = 0

    # 자신의 전 부분들과 비교 (range(i)는 자동으로 i-1까지 반복되기 때문에 자기와 비교하지 않음)
    for j in range(i):
      current_pos = i
      look_pos = j
      length = 0

      # 현재 위치가 데이터 밖으로 나갈려고 하면 취소
      # 비교 위치는 지금보다 전에 있어야만 함
      # loseless compression이기 때문에 문자가 같아야 함
      # 세 조건이 모두 충족될 경우 압축에 추가될 수 있음
      while (current_pos < len(data) and look_pos < i and data[current_pos] == data[look_pos]):
        length += 1
        current_pos += 1
        look_pos += 1

      # 자기 앞에 있는 문자열도 압축 가능하지 않을 때만 추가해야 압축률이 높아짐
      if length > max_length:
        max_length = length
        distance = i - j

    # 길이는 최소 1일 수밖에 없음
    max_length = max(1, max_length)
    # 계산 후 결과에 추가
    result.append(SlidingWindowedByte(distance, max_length, data[i:i+max_length]))
    # 압축된 부분의 길이만큼 이동해서 이미 본 부분은 건너뛰기함
    i += max_length
  return result

def decode_sliding_window(data: list[SlidingWindowedByte]) -> bytes:
  result = bytearray()
  for item in data:
    if item.distance == 0:
      # 아이템 거리가 0이라면 압축되지 않은 것이기 때문에 그냥 추가
      result.extend(item.value)
    else:
      # 전 부분에 거리를 빼서 시작 위치를 찾음
      start = len(result) - item.distance
      for i in range(item.length):
        # 길이만큼 추가
        result.append(result[start + i])

  return bytes(result)

def compress(file_path: str, output_path: str):
  raise NotImplementedError

def decompress(file_path: str, output_path: str):
  raise NotImplementedError

if __name__ == '__main__':
  sld = sliding_window(b'lzmaslidingtestforbytes')
  print(sld)
  print(f"res: {decode_sliding_window(sld)}")
  frq = build_frequency_table(sld)
  by = convert_to_bytes(sld)
  print(frq)
  enc = encode_range(frq, by)
  print(enc, by)
  dec = decode_range(frq, enc, len(by))
  print(dec)
