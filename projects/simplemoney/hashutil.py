from pathlib import Path
from hashlib import sha3_512
from sys import argv

salt = Path('./salt').read_bytes()
print(argv[1])
print(sha3_512(argv[1].encode('utf8') + salt).hexdigest())

