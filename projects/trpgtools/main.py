from gradio import Blocks, Textbox, Button
from secrets import SystemRandom

strings = ["쥐", "9급 해결사", "말단 조직원", "뒷골목 잡범", "뒷골목 도축업자", "8급 해결사", "조직원", "7급 해결사"]

def speedy_and_healthy(*args, **kwargs):
    args = list(args)
    args.extend(list(kwargs.values()))
    args = list(map(int, args))
    a = "name-number: healthdice-speeddice\n"
    for i, i2 in enumerate(args):
        b = []
        c = []
        if i >= 8:
            raise ValueError("too many")
        if i2 == 0:
            continue
        sdices = 1
        if i >= 6:
            hdices = 3
            sdices = 2
        elif i >= 4:
            hdices = 3
        elif i >= 2:
            hdices = 2
        else:
            hdices = 1
        for _ in range(i2):
            b.append(map(str, [SystemRandom().randint(1, 6) for _ in range(sdices)]))
            c.append(map(str, [SystemRandom().randint(1, 8) for _ in range(hdices)]))
        for i3, i4 in enumerate(zip(b, c)):
            a += f"{strings[i]}-{i3+1}: {','.join(i4[1])}-{','.join(i4[0])}\n"
    return a

with Blocks() as app:
    clist = [Textbox(0, label=f"{strings[i]}") for i in range(8)]
    submit = Button("제출")
    output = Textbox(label="Output")
    submit.click(fn=speedy_and_healthy, inputs=clist, outputs=output)

app.launch(share=True)
