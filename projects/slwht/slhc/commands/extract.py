from zipfile import ZipFile
from requests import get
from os.path import isdir
from os import makedirs, urandom, remove
from binascii import hexlify

def main(url: str, path: str = "workspace"):
    """
    Downloads a zip file from the given URL and extracts its contents to the
    specified directory.

    Args:
        url (str): The URL of the zip file to download.
        path (str, optional): The directory to extract the contents to.
            Defaults to "workspace".

    Returns:
        None
    """
    zip_file = get(url).content
    filename = hexlify(urandom(10)).decode()
    if not isdir(path):
        makedirs(path)
    with open(f"{path}/{filename}.zip", "wb") as f:
        f.write(zip_file)
    with ZipFile(f"{path}/{filename}.zip", 'r') as zip_ref:
        zip_ref.extractall(f"{path}")
    remove(f"{path}/{filename}.zip")
