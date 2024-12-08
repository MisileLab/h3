from collections import Counter
from copy import deepcopy
from pathlib import Path
from tempfile import NamedTemporaryFile
from RangeEncoder import compress as t_compress, decompress as t_decompress
from LZ77 import LZ77Compressor
from sys import argv
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

def decode_from_bytes(data: bytes) -> list[SlidingWindowedByte]:
  result = []
  offset = 0
  while offset < len(data):
    # Unpack two integers for distance and length
    distance, length = struct.unpack_from('ii', data, offset)
    offset += struct.calcsize('ii')

    # Find the null-terminated string
    value_start = offset
    value_end = data.find(0, value_start)  # Find the null byte
    if value_end == -1:
        raise ValueError("Malformed data: Null byte separator not found")

    # Extract the bytes
    value = data[value_start:value_end]
    offset = value_end + 1  # Move past the null byte

    # Create a SlidingWindowedByte instance and add it to the result
    result.append(SlidingWindowedByte(distance, length, value))
  return result

SCALE = 10**8

def build_possibility(sliding: bytes) -> dict[int, int]:
  c = Counter(sliding)
  total = sum(c.values())
  kv = {k: (c[k] * SCALE) // total for k in c}
  print(kv)
  if sum(kv.values()) == SCALE:
    kv[list(kv.keys())[-1]] += 1
  return kv

def encode_range(possibility: dict[int, int], values: bytes) -> int:
  possibility = deepcopy(possibility)
  ranges = {0: possibility[0]}
  del possibility[0]
  for k, v in possibility.items():
    val = max(ranges.values()) + v
    ranges[k] = val
  ranges[list(ranges.keys())[-1]] = SCALE

  r = SCALE
  low = 0
  for i in values:
    for k, v in ranges.items():
      ranges[k] = min(v * r // SCALE + low, SCALE)
    low = ranges[i]
    lr = list(ranges.keys())
    if lr.index(i) == len(lr) - 1:
      r = SCALE - low
    else:
      r = ranges[lr[lr.index(i) + 1]] - low
  return min(ranges.values())

def decode_range(possibility: dict[int, int], value: int, length: int) -> bytes:
  possibility = deepcopy(possibility)
  ranges = {0: possibility[0]}
  del possibility[0]
  for k, v in possibility.items():
    val = max(ranges.values()) + v
    ranges[k] = val
  ranges[list(ranges.keys())[-1]] = SCALE

  values = bytearray()
  r = SCALE
  low = 0
  for _ in range(length):
    for k, v in ranges.items():
      ranges[k] = v * r // SCALE + low
    for k, v in ranges.items():
      if value <= v:
        values.append(k)
        low = ranges[k]
        lr = list(ranges.keys())
        if lr.index(k) == len(lr) - 1:
          r = SCALE - low
        else:
          r = ranges[lr[lr.index(k) + 1]] - low
        break
    else:
      raise ValueError()
  return bytes(values)

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
  compressor = LZ77Compressor()
  with NamedTemporaryFile() as f:
    compressor.compress(file_path, f.name)
    t_compress(f.name, output_path)

def decompress(file_path: str, output_path: str):
  compressor = LZ77Compressor()
  with NamedTemporaryFile() as f:
    t_decompress(file_path, f.name)
    compressor.decompress(f.name, output_path)

if __name__ == '__main__':
  if len(argv) < 4:
    print("python tlzma_main.py [compress/decompress] [input_path] [output_path]")
    exit()
  if argv[1] == "compress":
    compress(argv[2], argv[3])
  else:
    decompress(argv[2], argv[3])
