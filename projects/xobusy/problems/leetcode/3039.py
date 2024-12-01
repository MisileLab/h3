from collections import Counter

class Solution:
  def lastNonEmptyString(self, s: str) -> str:
    c = Counter(reversed(s))
    mc = max(c.values())
    return ''.join(i for i in reversed(c) if c[i] == mc)
