from typing import final
from httpx import request

from os import getenv

mullvad_account_num = getenv("account_num")
if mullvad_account_num is None:
  print("mullvad account number not found in envrionment variable")
  print("please set account_num envrionment variable")
  exit(1)



