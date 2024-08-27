from manim import *

from secrets import SystemRandom

def create_textbox(rect: Rectangle, value: str) -> VGroup:
  res = VGroup()
  txt = Text(value).move_to(rect.get_center())
  res.add(rect, txt)
  return res

def get_non_duplicated_list(elements: int = 10) -> list[int]:
  rvalue = [SystemRandom().randint(1, 100) for _ in range(10)]
  while list(sorted(dict.fromkeys(rvalue))) != list(sorted(rvalue)):
    rvalue = [SystemRandom().randint(1, 100) for _ in range(10)]
  return rvalue

def list_to_vgroup(elements: list[int]) -> list[VGroup]:
  vgroups = []
  for i in elements:
    rect = Rectangle(height=1, width=1)
    vgroups.append(create_textbox(rect, str(i)))
  return vgroups

def move_vgroup(elements: list[VGroup]) -> list[Create]:
  elements[0].set_x(-5).set_y(0)
  for i in range(1, len(elements)):
    elements[i].next_to(elements[i-1], RIGHT)
    logger.debug(f"{elements[i].get_x()} {elements[i].get_y()}")
  return [Create(i) for i in elements]

def set_fill(vgroup: VGroup, color: str):
  return vgroup.submobjects[0].animate.set_fill(color, opacity=0.5)

def swap(vgroup1: VMobject, vgroup2: VMobject):
  a = vgroup1.get_center()
  b = vgroup2.get_center()
  return [vgroup1.animate.move_to(b), vgroup2.animate.move_to(a)]

