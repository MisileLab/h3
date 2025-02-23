from dataclasses import dataclass
from math import ceil
from os import getenv
from time import sleep
from re import compile, match
from json import dumps
from asyncio import run
from typing import Any

from selenium.webdriver import Firefox, FirefoxOptions
from selenium.webdriver.common.by import By
from httpx import USE_CLIENT_DEFAULT, BasicAuth, Client

rpc_endpoint = getenv("rpc_endpoint", "http://localhost:18088/json_rpc")
rpc_username = getenv("rpc_username")
rpc_password = getenv("rpc_password")
account_index = int(getenv("account_index", 0))
subaddress_index = int(getenv("subaddress_index", 0))
monero_regex = compile(r'/4[0-9AB][1-9A-HJ-NP-Za-km-z]{93}$/g')

async def rpc(client: Client, method: str, params: dict[str, Any]) -> dict[str, Any]: # pyright: ignore[reportExplicitAny]
  req = client.post(
    rpc_endpoint,
    data={
      "jsonrpc": "2.0",
      "id": "0",
      "method": method,
      "inputs": dumps(params)
    },
    auth=BasicAuth(rpc_username, rpc_password) if rpc_username and rpc_password else USE_CLIENT_DEFAULT
  )
  _ = req.raise_for_status()
  return req.json() # pyright: ignore[reportAny]

def from_piconero(amount: int) -> float:
  return amount / 10 ** 12

def to_piconero(amount: float) -> int:
  return ceil(amount * 10 ** 12)

@dataclass
class Transaction:
  address: str
  amount: float

async def get_mullvad_transaction(account_number: str) -> Transaction:
  options = FirefoxOptions()
  options.add_argument('--headless') # pyright: ignore[reportUnknownMemberType]

  driver = Firefox()
  driver.get("https://mullvad.net/en/account/login")
  assert driver.title == "Log in | Mullvad VPN"
  login_form = driver.find_elements(By.CSS_SELECTOR, 'form')[0]
  login_form_input = login_form.find_element(By.NAME, 'account_number') # pyright: ignore[reportUnknownMemberType]
  assert login_form_input.get_attribute('placeholder') == 'Enter your account number' # pyright: ignore[reportUnknownMemberType]
  assert login_form_input.get_attribute('type') == 'password' # pyright: ignore[reportUnknownMemberType]
  login_form_button = login_form.find_element(By.CSS_SELECTOR, "[type='submit']") # pyright: ignore[reportUnknownMemberType]
  login_form_input.clear()
  login_form_input.send_keys(account_number)
  login_form_button.click()
  while driver.current_url != 'https://mullvad.net/en/account':
    print(f"waiting redirect from {driver.current_url}")
    sleep(1)
  driver.get('https://mullvad.net/en/account/payment/monero')
  while driver.find_element(
    By.CLASS_NAME,
    'payment'
  ).find_element(By.CSS_SELECTOR, 'h2').text != 'Pay with Monero': # pyright: ignore[reportUnknownMemberType]
    print("waiting render")
    sleep(1)
  create_address = driver.find_element(
    By.CLASS_NAME, 'create'
  ).find_element(By.CSS_SELECTOR, "[type='submit']") # pyright: ignore[reportUnknownMemberType]
  assert create_address.text == 'Create a one-time payment address'
  create_address.click()
  addresses = driver.find_elements(By.CSS_SELECTOR, "[data-cy='address-field']")
  while len(addresses) == 0:
    addresses = driver.find_elements(By.CSS_SELECTOR, "[data-cy='address-field']")
    if len(addresses) > 0:
      print("payment address generated")
      break
    print("sleeping because mullvad generating payment address")
    sleep(1)
  address = addresses[0].text
  print(address)
  amount = float(driver.find_element(By.CSS_SELECTOR, "[data-cy='amount-field']").text)
  print(amount)
  driver.close()
  return Transaction(address, amount)

async def payment(transaction: Transaction):
  assert match(monero_regex, transaction.address) is not None
  print(f"I'll send {transaction.amount} XMR to {transaction.address} from {account_index}th account")
  if input("If this transaction is correct, please input 'correct'. ") != "correct":
    print("transaction aborted")
    exit(1)
  with Client() as c:
    balance = from_piconero((await rpc(c, "get_balance", {
      "account_index": account_index,
      "address_index": [subaddress_index]
    }))["result"]["unlocked_balance"]) # pyright: ignore[reportAny]
    assert balance > transaction.amount
    res = await rpc(c, "transfer", {
      "destinations": [{"amount": to_piconero(transaction.amount), "address": transaction.address}],
      "account_index": account_index,
      "subaddr_indicies": [subaddress_index],
      "priority": 0
    })
    fee: float = from_piconero(res["fee"]) # pyright: ignore[reportAny]
    total: float = from_piconero(res["amount"]) # pyright: ignore[reportAny]
    tx_hash: str = res["tx_hash"]
    print(f"transaction hash: {tx_hash}")
    print(f"fee: {fee}")
    print(f"total: {total}")


async def main():
  mullvad_account_num = getenv("account_num")
  if mullvad_account_num is None:
    print("mullvad account number not found in envrionment variable")
    print("please set account_num envrionment variable")
    exit(1)
  transaction = await get_mullvad_transaction(mullvad_account_num)
  await payment(transaction)

if __name__ == "__main__":
  run(main())

