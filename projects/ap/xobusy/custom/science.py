from unicode import join_jamos

a = [
    'CTTTTTTAATTATCTTCCGTTAGATAAGTAGACTGACGTTGGTTAGACGTTGACTGATACGGACGAGACGTTGACATT',
    'AATCGATTAGGATTGGACTGACGGTTGGGAGTAGGGCGAAGAGTTAGGTAGGTAGACTGGCGTTGATTGGACGTTGACTGATACGGACGAGACGTTGACATT',
    'TGGGCAAGGGGTTTATCTTCCGTCAGCGACTTGCTCACAGTGGACGTCGACTGACGAAGTGTCCTGCGGGACTCCGTCGACATC',
    'TCTTTTTAATCCCAGGGACCAGACCGAAATCTATTAAGGTCTGTTAGGTCTGCAGGGGGTTTATCTTCCGTTAGACGATAAGTAGACTGACGTTGGTTAGACGTTGACTGATACGGACGAGACGTTGACATT',
    'GACGTAGACGGTCGATCTCTTTTATCTGTCTTTGGGCGCGAATCCGTCAGAGAAGCAAGTCTTCCATCTTGGTTGGACGTCGACGGGCGTGAGGGACGAGACGTCGACATC'
]
c = {"A": "U", "T": "A", "C": "G", "G": "C", " ": " "}
b = ["".join(c[i2] for i2 in i) for i in a]
print('ㅁㄴㅇㅁㅇㄴ\n'.join(b))
print('-------------------------')
d = {
    ("UUU", "UUC"): "ㅍ",
    ("UUA", "UUG", "CUU", "CUC", "CUA"): "ㄱ",
    ("CUG"): "ㄴ",
    ("AUU", "AUC", "AUA"): "ㅂ",
    ("AUG"): "ㅛ",
    ("GUU", "GUC", "GUA", "GUG"): "ㅔ",
    ("UCU", "UCC", "UCA", "UCG"): "ㄹ",
    ("CCU", "CCC"): "ㅅ",
    ("CCA", "CCG"): "ㄷ",
    ("ACU", "ACC"): "ㅎ",
    ("ACA", "ACG"): "ㅒ",
    ("GCU", "GCC"): "ㅗ",
    ("GCA", "GCG"): "ㅐ",
    ("UAU", "UAC"): "ㅠ",
    ("UAA", "UAG"): "?",
    ("CAU", "CAC"): "ㅜ",
    ("CAA", "CAG"): "ㅡ",
    ("AAU", "AAC"): "ㅏ",
    ("AAA", "AAG"): "ㅣ",
    ("GAU", "GAC"): "ㅁ",
    ("GAA", "GAG"): "ㅈ",
    ("UGU", "UGC"): "ㅊ",
    ("UGA", "UGG"): "ㅋ",
    ("CGU", "CGC"): "ㅕ",
    ("CGA", "CGG"): "ㅖ",
    ("AGU", "AGC"): "ㅌ",
    ("AGA", "AGG"): "ㅇ",
    ("GGU", "GGC"): "ㅓ",
    ("GGA", "GGG"): "ㅑ"
}

def e(f: str):
    _buffer, g = "", ""
    for i in f:
        if i == ' ':
            g += ' '
            continue
        _buffer += i
        if len(_buffer) == 3:
            for j in d.keys():
                if _buffer in j:
                    g += d[j]
                    break
            _buffer = ""
    return g

for i in b:
    print(join_jamos(e(i)))
