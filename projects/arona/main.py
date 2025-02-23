from os import getenv
from time import sleep

from selenium.webdriver import Firefox, FirefoxOptions
from selenium.webdriver.common.by import By

mullvad_account_num = getenv("account_num")
if mullvad_account_num is None:
  print("mullvad account number not found in envrionment variable")
  print("please set account_num envrionment variable")
  exit(1)

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
login_form_input.send_keys(mullvad_account_num)
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

