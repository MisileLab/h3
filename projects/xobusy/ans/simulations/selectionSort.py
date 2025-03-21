from manim import *

from lib import *

class SelectionSort(Scene):
  def create_textbox(self, rect: Rectangle, value: str) -> VGroup:
    res = VGroup()
    txt = Text(value).move_to(rect.get_center())
    res.add(rect, txt)
    return res

  def construct(self):
    self.elements = get_non_duplicated_list()
    self.squares: list[VGroup] = list_to_vgroup(self.elements)
    self.play(*[move_vgroup(self.squares)])
    for i in range(len(self.squares)-1):
      self.play(set_fill(self.squares[i], 'blue')) # type: ignore
      self.wait(0.05)
      min_num = i
      for j in range(i+1, len(self.elements)):
        self.play(set_fill(self.squares[j], 'yellow')) # type: ignore
        if self.elements[min_num] > self.elements[j]:
          animation_list = []
          if min_num != i:
            animation_list.append(set_fill(self.squares[min_num], 'yellow'))
          if min_num != j:
            animation_list.append(set_fill(self.squares[j], 'red'))
          self.play(*animation_list)
          min_num = j
      tmp = self.elements[i]
      self.elements[i] = self.elements[min_num]
      self.elements[min_num] = tmp
      self.play(*[swap(self.squares[i], self.squares[min_num])]) # type: ignore
      t = self.squares[i]
      self.squares[i] = self.squares[min_num]
      self.squares[min_num] = t
      del t
      self.play(*[j.submobjects[0].animate.set_fill('black', opacity=1) for j in self.squares[i:]]) # type: ignore
      self.play(set_fill(self.squares[i], 'green')) # type: ignore
    self.play(set_fill(self.squares[-1], 'green')) # type: ignore
    self.wait(2)


