# This is a generated file! Please edit source .ksy file and use kaitai-struct-compiler to rebuild

import kaitaistruct
from kaitaistruct import KaitaiStruct, KaitaiStream, BytesIO


if getattr(kaitaistruct, 'API_VERSION', (0, 9)) < (0, 9):
    raise Exception("Incompatible Kaitai Struct Python API: 0.9 or later is required, but you have %s" % (kaitaistruct.__version__))

class Tlzma(KaitaiStruct):
    def __init__(self, _io, _parent=None, _root=None):
        self._io = _io
        self._parent = _parent
        self._root = _root if _root else self
        self._read()

    def _read(self):
        self.header = Tlzma.Header(self._io, self, self._root)
        self.value = self._io.read_f8le()
        self.frequency = []
        i = 0
        while True:
            _ = Tlzma.Frequency(self._io, self, self._root)
            self.frequency.append(_)
            if self._io.is_eof():
                break
            i += 1

    class Header(KaitaiStruct):
        def __init__(self, _io, _parent=None, _root=None):
            self._io = _io
            self._parent = _parent
            self._root = _root if _root else self
            self._read()

        def _read(self):
            self.magic = self._io.read_bytes(5)
            if not self.magic == b"\x54\x4C\x5A\x4D\x41":
                raise kaitaistruct.ValidationNotEqualError(b"\x54\x4C\x5A\x4D\x41", self.magic, self._io, u"/types/header/seq/0")


    class Frequency(KaitaiStruct):
        def __init__(self, _io, _parent=None, _root=None):
            self._io = _io
            self._parent = _parent
            self._root = _root if _root else self
            self._read()

        def _read(self):
            self.distance = self._io.read_u4le()
            self.length = self._io.read_u4le()
            self.value = (self._io.read_bytes(self.length)).decode(u"UTF-8")



