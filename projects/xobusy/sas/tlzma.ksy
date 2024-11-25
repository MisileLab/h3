meta:
  id: tlzma
  file-extension: tlzma
  endian: le
seq:
  - id: header
    type: header
  - id: value
    type: f8
  - id: frequency
    type: frequency
    repeat: until
    repeat-until: _io.eof
types:
  header:
    seq:
      - id: magic
        contents: "TLZMA"
        size: 5
  frequency:
    seq:
      - id: distance
        type: u4
      - id: length
        type: u4
      - id: value
        type: str
        size: length
        encoding: UTF-8