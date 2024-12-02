from pathlib import Path
from tempfile import NamedTemporaryFile
from RangeEncoder import decompress as t_decompress, compress as t_compress
from bson import loads, dumps

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

def to_bson(sliding_windows: list[SlidingWindowedByte]) -> bytes:
  """
  Encode list of SlidingWindowedByte objects to BSON format
  """

  doc_list = [
    {
      'distance': sw.distance,
      'length': sw.length,
      'value': sw.value
    }
    for sw in sliding_windows
  ]

  doc = {'sliding_windows': doc_list}
  return dumps(doc)

def from_bson(bson_data: bytes) -> list[SlidingWindowedByte]:
  """
  Decode BSON data to list of SlidingWindowedByte objects
  """

  doc = loads(bson_data)
  return [
    SlidingWindowedByte(
      distance=item['distance'],
      length=item['length'],
      value=item['value']
    )
    for item in doc['sliding_windows'] # type: ignore
  ]

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
  with NamedTemporaryFile() as f:
    d = dumps({'a': [i.to_tuple() for i in sliding_window(Path(file_path).read_bytes())]})
    Path(f.name).write_bytes(d)
    t_compress(f.name, output_path)
    return d

def decompress(file_path: str, output_path: str, d: bytes):
  with NamedTemporaryFile() as f:
    t_decompress(file_path, f.name)
    assert d == Path(f.name).read_bytes()
    l: dict[str, list[tuple[int, int, bytes]]] = loads(Path(f.name).read_bytes()) # type: ignore
    Path(output_path).write_bytes(decode_sliding_window([SlidingWindowedByte(*i) for i in l['a']]))

if __name__ == '__main__':
  d = compress("./test", "./test.tlzma")
  decompress("./test.tlzma", "./test.py.org", d)
