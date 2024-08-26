from manim import *
from loguru import logger

from secrets import SystemRandom

class SelectionSort(Scene):
  def create_textbox(self, rect: Rectangle, value: str) -> VGroup:
    res = VGroup()
    txt = Text(value).move_to(rect.get_center())
    res.add(rect, txt)
    return res

  def construct(self):
    self.elements = [SystemRandom().randint(1, 100) for _ in range(10)]
    while list(sorted(list(dict.fromkeys(self.elements)))) != list(sorted(self.elements)):
      self.elements = [SystemRandom().randint(1, 100) for _ in range(10)]
    self.squares: list[VGroup] = []
    for i in self.elements:
      rect = Rectangle(height=1, width=1)
      self.squares.append(self.create_textbox(rect, str(i)))
    for i in range(len(self.squares)):
      if i != 0:
        self.squares[i].next_to(self.squares[i-1], RIGHT)
      else:
        self.squares[i].set_x(-5).set_y(0)
      logger.debug(f"{self.squares[i].get_x()} {self.squares[i].get_y()}")
    self.play(*[Create(i) for i in self.squares])
    for i in range(len(self.squares)-1):
      self.play(self.squares[i].submobjects[0].animate.set_fill('blue', opacity=0.5)) # type: ignore
      self.wait(0.05)
      min_num = i
      for j in range(i+1, len(self.elements)):
        self.play(self.squares[j].submobjects[0].animate.set_fill('yellow', opacity=0.5)) # type: ignore
        if self.elements[min_num] > self.elements[j]:
          animation_list = []
          if min_num != i:
            animation_list.append(self.squares[min_num].submobjects[0].animate.set_fill('yellow', opacity=0.5))
          if min_num != j:
            animation_list.append(self.squares[j].submobjects[0].animate.set_fill('red', opacity=0.5))
          self.play(*animation_list)
          min_num = j
      tmp = self.elements[i]
      self.elements[i] = self.elements[min_num]
      self.elements[min_num] = tmp
      self.play(*[j.submobjects[0].animate.set_fill('black', opacity=1) for j in self.squares]) # type: ignore
      self.play(Transform(self.squares[i].submobjects[1], Text(str(self.elements[i])).move_to(self.squares[i].submobjects[0].get_center())),
                Transform(self.squares[min_num].submobjects[1], Text(str(self.elements[min_num])).move_to(self.squares[min_num].submobjects[0].get_center())))
