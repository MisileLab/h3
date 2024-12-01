class Solution(object):
  def divideString(self, s: str, k: int, fill: str) -> list[str]:
    res = []
    l = 0
    ls = len(s)
    isdiv = ls % k == 0
    while l*k < ls:
      if not isdiv and ls < (l+1)*k:
        res.append(s[l*k:] + fill * (k - (ls % k)))
        return res
      res.append(s[l*k:(l+1)*k])
      l += 1
    return res

