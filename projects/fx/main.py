from manim import *

def remove_all_objects(self: Scene):
    self.play(
            *[FadeOut(mob)for mob in self.mobjects]
    )

class Main(Scene):
    def construct(self):
        #formula1(self)
        formula2(self)
        self.wait(1)

# (a+b)^2=a+2ab+b^2, a=2, b=1
def formula1(self: Scene):
    res = Square(side_length=3).move_to(LEFT*4).set_fill(BLUE, opacity=0.7)
    text = Tex("$(a+b)^2$").next_to(res, DOWN)
    self.play(Create(res), FadeIn(text))
    res2 = [
        Square(side_length=2, color=RED),
        Rectangle(height=2, width=1, color=BLUE),
        Rectangle(height=1, width=2, color=BLUE),
        Square(side_length=1, color=YELLOW)
    ]
    res2[0].move_to(UP*0.5)
    res2[1].next_to(res2[0], RIGHT)
    res2[2].next_to(res2[0], DOWN)
    res2[3].next_to(res2[1], DOWN)
    self.play(Create(res2[0]))
    self.play(Create(res2[1]), Create(res2[2]))
    self.play(Create(res2[3]))
    texts = [
        Tex("$b$").next_to(res2[3], DOWN),
        Tex("$a$").next_to(res2[2], DOWN)
    ]
    restexts = [
        Tex("$a^2$").move_to(res2[0].get_center()),
        Tex("$ab$").move_to(res2[1].get_center()),
        Tex("$ab$").move_to(res2[2].get_center()),
        Tex("$b^2$").move_to(res2[3].get_center())
    ]
    self.play(FadeIn(texts[0]), FadeIn(texts[1]), FadeIn(restexts[0]), FadeIn(restexts[1]), FadeIn(restexts[2]), FadeIn(restexts[3]))
    groups = [
        Group(restexts[0], res2[0]),
        Group(restexts[1], res2[1]),
        Group(restexts[2], res2[2], texts[1]),
        Group(restexts[3], res2[3], texts[0])
    ]
    realres = Tex("$a^2$").next_to(texts[1], DOWN*2)
    self.play(Transform(groups[0], realres))
    realres2 = Tex("$+ab$").next_to(realres, RIGHT*0.5)
    self.wait(1)
    self.play(Transform(groups[1], realres2))
    realres3 = Tex("$+ab$").next_to(realres2, RIGHT*0.5)
    self.wait(1)
    self.play(Transform(groups[2], realres3))
    realres4 = Tex("$+b^2$").next_to(realres3, RIGHT*0.5)
    self.wait(1)
    self.play(Transform(groups[3], realres4))
    res = Tex("$a^2+2ab+b^2$")
    res.move_to(text.get_center())
    self.play(FadeOut(text), Transform(Group(realres, realres2, realres3, realres4), res))
    self.wait(2)
    remove_all_objects(self)

# (a+b)(a-b) = a^2 - b^2; a=3, b=1
def formula2(self: Scene):
    res = Square(side_length=3).move_to(LEFT*4).set_fill(BLUE, opacity=0.7)
    text = Tex("$(a+b)(a-b)$").next_to(res, DOWN)
    self.play(Create(res), FadeIn(text))
    res2 = [
        Square(side_length=3, color=RED),
        Rectangle(height=3, width=1, color=BLUE),
        Rectangle(height=1, width=3, color=BLUE),
        Square(side_length=1, color=YELLOW)
    ]
    res2[0].move_to(UP*0.5)
    res2[1].next_to(res2[0], RIGHT)
    res2[2].next_to(res2[0], DOWN)
    res2[3].next_to(res2[1], DOWN)
    self.play(Create(res2[0]))
    self.play(Create(res2[1]), Create(res2[2]))
    self.play(Create(res2[3]))
    texts = [
        Tex("$-b$").next_to(res2[3], DOWN),
        Tex("$a$").next_to(res2[2], DOWN),
        Tex("$b$").next_to(res2[3], RIGHT)
    ]
    restexts = [
        Tex("$a^2$").move_to(res2[0].get_center()),
        Tex("$-ab$").move_to(res2[1].get_center()),
        Tex("$ab$").move_to(res2[2].get_center()),
        Tex("$-b^2$").move_to(res2[3].get_center())
    ]
    self.play(FadeIn(texts[0]), FadeIn(texts[1]), FadeIn(texts[2]), FadeIn(restexts[0]), FadeIn(restexts[1]), FadeIn(restexts[2]), FadeIn(restexts[3]))
    groups = [
        Group(restexts[0], res2[0]),
        Group(restexts[1], res2[1]),
        Group(restexts[2], res2[2], texts[1]),
        Group(restexts[3], res2[3], texts[0], texts[2])
    ]
    realres = Tex("$a^2$").next_to(texts[1], DOWN*2)
    self.play(Transform(groups[0], realres))
    realres2 = Tex("$-ab$").next_to(realres, RIGHT*0.5)
    self.wait(1)
    self.play(Transform(groups[1], realres2))
    realres3 = Tex("$+ab$").next_to(realres2, RIGHT*0.5)
    self.wait(1)
    self.play(Transform(groups[2], realres3))
    realres4 = Tex("$-b^2$").next_to(realres3, RIGHT*0.5)
    self.wait(1)
    self.play(Transform(groups[3], realres4))
    res = Tex("$a^2-b^2$")
    res.move_to(text.get_center())
    self.play(FadeOut(text), Transform(Group(realres, realres2, realres3, realres4), res))
    self.wait(2)
    remove_all_objects(self)
