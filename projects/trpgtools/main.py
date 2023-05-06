from gradio import Blocks, Textbox, Button
from secrets import SystemRandom
from json import load

data = load(open("data.json", "r", encoding="utf-8"))
strings = [i["name"] for i in data]

def speedy_and_healthy(*args, **kwargs):
    args = list(args)
    args.extend(list(kwargs.values()))
    args = list(map(int, args))
    a = "name-number: healthdice-speeddice\n"
    for i, i2 in enumerate(args):
        b = []
        c = []
        sdices = data[i]["SD"]
        hdices = data[i]["HD"]
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
