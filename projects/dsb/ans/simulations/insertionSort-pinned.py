from manim import *

from lib import *

class InsertionSortPinned(Scene):
  def create_textbox(self, rect: Rectangle, value: str) -> VGroup:
    res = VGroup()
    txt = Text(value).move_to(rect.get_center())
    res.add(rect, txt)
    return res

  def construct(self):
    self.elements = [3,7,9,4,1,6]
    self.squares: list[VGroup] = list_to_vgroup(self.elements)
    self.play(*[move_vgroup(self.squares)])
    for i in range(1, len(self.elements)):
      self.play(*[set_fill(j, 'black') for j in self.squares])
      k = i
      self.play(set_fill(self.squares[i], 'blue'))
      for j in range(i-1, -1, -1):
        self.play(set_fill(self.squares[j], 'yellow'))
        if self.elements[i]<self.elements[j]:
          animations = [set_fill(self.squares[j], 'red')]
          if k != i:
            animations.append(set_fill(self.squares[k], 'yellow'))
          self.play(*[animations])
          k = j
        else:
          break
      logger.info(f"{k}, {i}, {self.elements}")
      if k != i:
        t = self.elements[i]
        del self.elements[i]
        self.elements.insert(k, t)
        t = self.squares[i]
        del self.squares[i]
        self.squares.insert(k, t)
        self.play(*realign_vgroup(self.squares))
        del t
    self.play(*[set_fill(k, 'green') for k in self.squares])
    self.wait(2)

