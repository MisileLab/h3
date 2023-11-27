from shutil import get_terminal_size
from time import strftime, sleep


def print_line_vcentred(s):
    centre_line = int(get_terminal_size().lines / 2)
    print('\n' * centre_line, s.center(get_terminal_size().columns),
          '\n' * centre_line)


while True:
    print_line_vcentred(strftime('%H:%M:%S'))
    sleep(1)
