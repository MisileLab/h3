from misilelibpy import cls
from os import _exit
from modules import module1 as md1

def main():
    while True:
        print("Percentage Calculator With Game v0.0.1-rolling")
        print("[1] Scp: SL Loot Calculator")
        print("[q] quit")
        a = input()
        if a == "q":
            _exit(0)
        else:
            try:
                a = int(a)
            except ValueError:
                print("Invalid input")
                print("Enter to continue")
                input()
            else:
                if a == 1:
                    md1.scp_sl_loot()
                input()
        cls()

if __name__ == "__main__":
    main()
