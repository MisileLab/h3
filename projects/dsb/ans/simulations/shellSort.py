from manim import *
from loguru import logger

from lib import *

from math import floor

class ShellSort(Scene):
  def create_textbox(self, rect: Rectangle, value: str) -> VGroup:
    res = VGroup()
    txt = Text(value).move_to(rect.get_center())
    res.add(rect, txt)
    return res

  def construct(self):
    colors = [BLUE_C, RED_C, PURPLE_C, YELLOW_C, GREEN_C]
    self.elements = get_non_duplicated_list()
    self.squares: list[VGroup] = list_to_vgroup(self.elements)
    self.play(*[move_vgroup(self.squares)])
    h = len(self.elements)
    while h > 1:
      h = floor(h / 2)
      if h % 2 == 0:
        h += 1
      logger.info(h)
      for i in range(h):
        self.play(*[set_fill(self.squares[j], colors[i], opacity=0.8, index=1) for j in range(i, len(self.elements), h)]) # type: ignore
      for i in range(h):
        for j in range(i+h, len(self.elements), h):
          k = self.elements[j]
          k_sqr = self.squares[j]
          l = j - h
          print(l, i, k)
          early_return = False
          if l < i or self.elements[l] <= k:
            self.play(*[set_fill(a, 'yellow') for a in [k_sqr, self.squares[l]]])
            early_return = True
          while l >= i and self.elements[l] > k:
            self.play(*[set_fill(a, 'yellow') for a in [k_sqr, self.squares[l]]])
            tmp = self.elements[l]
            self.elements[l] = self.elements[l+h]
            self.elements[l+h] = tmp
            del tmp
            self.play(*[set_fill(a, 'black') for a in [k_sqr, self.squares[l]]])
            tmp = self.squares[l]
            self.squares[l] = self.squares[l+h]
            self.squares[l+h] = tmp
            del tmp
            l -= h
          if not early_return:
            self.elements[l+h] = k
            self.squares[l+h] = k_sqr
            self.play(*realign_vgroup(self.squares))
          if early_return:
            self.play(*[set_fill(a, 'black') for a in self.squares])
        print("one iteration completed")
      print("two iterations completed")
      self.play(*[set_fill(j, 'white', opacity=1, index=1) for j in self.squares])
    self.play(*[set_fill(j, 'green') for j in self.squares])
    self.wait(2)
    print(self.elements)
