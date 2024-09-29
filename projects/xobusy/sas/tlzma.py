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
