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
    #self.elements = get_non_duplicated_list()
    self.elements = [10,4,32,9,5,3,20,94,99,30]
    self.squares: list[VGroup] = list_to_vgroup(self.elements)
    self.play(*[move_vgroup(self.squares)])
    self.subelements: list[list[int]] = []
    self.subsquares: list[list[VGroup]] = []
    k = len(self.elements)
    while True:
      self.subsquares = []
      self.subelements = []
      k = floor(k / 2)
      if k == 1:
        break
      if k % 2 == 0:
        k += 1
      logger.info(k)
      for i in range(0, k):
        logger.info(self.subsquares)
        logger.info(self.subelements)
        self.subsquares.append([])
        self.subelements.append([])
        for j in range(i, len(self.elements), k):
          self.subsquares[-1].append(self.squares[j])
          self.subelements[-1].append(self.elements[j])
        self.play(*[set_fill(j, colors[i], opacity=0.8, index=1) for j in self.subsquares[-1]]) # type: ignore
        for j in range(1, len(self.subelements[-1])):
          for v in range(j-1, 0, -1):
            logger.info(f"info: {self.subelements[-1][j]}, {self.subelements[-1][i]}")
            self.play(*[set_fill(self.subsquares[-1][l], 'yellow') for l in [j, v]])
            if self.subelements[-1][j]>self.subelements[-1][v]:
              logger.warning(v)
              tmp = self.subsquares[-1][v]
              del self.subsquares[-1][v]
              self.subsquares[-1].insert(v, tmp)
              del tmp
              self.play(*realign_vgroup(self.squares))
              self.play(*[set_fill(self.subsquares[-1][l], 'black') for l in [j, v]])
              break
            self.play(*[set_fill(self.subsquares[-1][l], 'black') for l in [j, v]])
      self.play(*[set_fill(j, 'white', opacity=1, index=1) for j in self.squares])
    self.play(*[set_fill(j, 'green') for j in self.squares])
    self.wait(2)

