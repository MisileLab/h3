from pathlib import Path
from hashlib import sha3_512

salt = Path('./salt').read_bytes()
pw = input()
print(pw)
print(sha3_512(pw.encode('utf8') + salt).hexdigest())

