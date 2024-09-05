from manim import *
from manim.typing import Point3D

from secrets import SystemRandom

def create_textbox(rect: Rectangle, value: str) -> VGroup:
  res = VGroup()
  txt = Text(value).move_to(rect.get_center())
  res.add(rect, txt)
  return res

def get_non_duplicated_list(elements: int = 10) -> list[int]:
  rvalue = [SystemRandom().randint(1, 100) for _ in range(elements)]
  while list(sorted(dict.fromkeys(rvalue))) != list(sorted(rvalue)):
    rvalue = [SystemRandom().randint(1, 100) for _ in range(elements)]
  return rvalue

def list_to_vgroup(elements: list[int]) -> list[VGroup]:
  vgroups = []
  for i in elements:
    rect = Rectangle(height=1, width=1)
    vgroups.append(create_textbox(rect, str(i)))
  return vgroups

def move_vgroup(elements: list[VGroup], base_x: float = -5, base_y: float = 0) -> list[Animation]:
  elements[0].set_x(base_x).set_y(base_y)
  for i in range(1, len(elements)):
    elements[i].next_to(elements[i-1], RIGHT)
    logger.info(f"{elements[i].get_x()} {elements[i].get_y()}")
  return [Create(i) for i in elements]

def realign_vgroup(elements: list[VGroup], base_x: float = -5, base_y: float = 0) -> list[Animation]:
  animations = [elements[0].animate.move_to((base_x, base_y, 0))]
  for i in range(1, len(elements)):
    animations.append(elements[i].animate.move_to((base_x+i*1.25, base_y, 0)))
    logger.info(f"{elements[i].get_x()} {elements[i].get_y()}")
  return animations # type: ignore

def set_fill(vgroup: VGroup, color: str | ManimColor, index: int = 0, opacity: float = 0.5) -> Animation:
  return vgroup.submobjects[index].animate.set_fill(color, opacity=opacity) # type: ignore

def swap(vgroup1: VMobject, vgroup2: VMobject) -> list[Animation]:
  a = vgroup1.get_center()
  b = vgroup2.get_center()
  return [vgroup1.animate.move_to(b), vgroup2.animate.move_to(a)] # type: ignore

def swap_value(l: list, index: int, index_2: int):
  tmp = l[index]
  l[index] = l[index_2]
  l[index_2] = tmp
  del tmp
    
