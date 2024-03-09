a = int(input())
itmo = [1995, 1998, 1999, 2001, 2002, 2003, 2004, 2005, 2009, 2010, 2011, 2012, 2014, 2015, 2016, 2017, 2019]
spbsu = [1996, 1997, 2000, 2007, 2008, 2013, 2018]

if a in itmo:
  print("ITMO")
elif a in spbsu:
  print("SPbSU")
else:
  print("PetrSU, ITMO")
