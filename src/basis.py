import sys
import numpy as np

def basiset(basis):
   """Return basis set dimensions"""

   lmax = {}
   nmax = {}

   if basis=="RI-ccpVQZ":

      lmax["H"] = 4
      lmax["C"] = 5
      lmax["N"] = 5
      lmax["O"] = 5

      # hydrogen
      nmax[("H",0)] = 4
      nmax[("H",1)] = 3
      nmax[("H",2)] = 3
      nmax[("H",3)] = 2
      nmax[("H",4)] = 1
      # carbon
      nmax[("C",0)] = 10
      nmax[("C",1)] = 7
      nmax[("C",2)] = 5
      nmax[("C",3)] = 3
      nmax[("C",4)] = 2
      nmax[("C",5)] = 1
      # nytrogen
      nmax[("N",0)] = 10
      nmax[("N",1)] = 7
      nmax[("N",2)] = 5
      nmax[("N",3)] = 3
      nmax[("N",4)] = 2
      nmax[("N",5)] = 1
      # oxygen
      nmax[("O",0)] = 10
      nmax[("O",1)] = 7
      nmax[("O",2)] = 5
      nmax[("O",3)] = 3
      nmax[("O",4)] = 2
      nmax[("O",5)] = 1
   
   if basis=="FHI-aims-first-tier":

      lmax["H"] = 2
      lmax["O"] = 4

      # hydrogen
      nmax[("H",0)] = 3
      nmax[("H",1)] = 2
      nmax[("H",2)] = 1
      # oxygen
      nmax[("O",0)] = 6
      nmax[("O",1)] = 5
      nmax[("O",2)] = 4
      nmax[("O",3)] = 3
      nmax[("O",4)] = 1
      
   if basis=="FHI-aims-third-tier":

      lmax["H"] = 5
      lmax["C"] = 5
      lmax["N"] = 5
      lmax["O"] = 5

      # hydrogen
      nmax[("H",0)] = 11
      nmax[("H",1)] = 9
      nmax[("H",2)] = 7
      nmax[("H",3)] = 5
      nmax[("H",4)] = 3
      nmax[("H",5)] = 2
      # oxygen
      nmax[("O",0)] = 10
      nmax[("O",1)] = 10
      nmax[("O",2)] = 9
      nmax[("O",3)] = 9
      nmax[("O",4)] = 8
      nmax[("O",5)] = 5

   if basis=="FHI-aims-min":

      lmax["H"] = 0
      lmax["O"] = 2
      lmax["Al"] = 2
      lmax["Si"] = 2

      nmax[("H",0)] = 1
      nmax[("O",0)] = 3
      nmax[("O",1)] = 2
      nmax[("O",2)] = 1
      nmax[("Al",0)] = 7
      nmax[("Al",1)] = 7
      nmax[("Al",2)] = 3
      nmax[("Si",0)] = 7
      nmax[("Si",1)] = 7
      nmax[("Si",2)] = 3

   return [lmax,nmax]
