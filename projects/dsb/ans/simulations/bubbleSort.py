from manim import *

from lib import *

class BubbleSort(Scene):
  def create_textbox(self, rect: Rectangle, value: str) -> VGroup:
    res = VGroup()
    txt = Text(value).move_to(rect.get_center())
    res.add(rect, txt)
    return res

  def construct(self):
    self.elements = get_non_duplicated_list()
    self.squares: list[VGroup] = list_to_vgroup(self.elements)
    self.play(*[move_vgroup(self.squares)])
    for i in range(len(self.elements)):
      swapped = False
      for j in range(len(self.elements)-1-i):
        self.play(*[set_fill(self.squares[j], 'yellow'), set_fill(self.squares[j+1], 'yellow')])
        if self.elements[j]>self.elements[j+1]:
          t = self.elements[j]
          self.elements[j] = self.elements[j+1]
          self.elements[j+1] = t
          self.play(swap(self.squares[j], self.squares[j+1]))
          t = self.squares[j]
          self.squares[j] = self.squares[j+1]
          self.squares[j+1] = t
          swapped = True
          del t
        self.play(*[set_fill(self.squares[j], 'black'), set_fill(self.squares[j+1], 'black')])
      if not swapped:
        break
      self.play(set_fill(self.squares[-(i+1)], 'green'))
    self.play(*[set_fill(i, 'green') for i in self.squares])
    self.wait(2)

