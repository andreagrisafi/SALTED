import sys
import numpy as np

def basiset(basis):
   """Return basis set dimensions"""

   lmax = {}
   nmax = {}


   if basis=="def2-SV(P)-JKFIT":
      lmax["H"] = 2
      lmax["He"] = 2
      lmax["Li"] = 3
      lmax["Be"] = 3
      lmax["B"] = 4
      lmax["C"] = 4
      lmax["N"] = 4
      lmax["O"] = 4
      lmax["F"] = 4
      lmax["Ne"] = 4
      lmax["Na"] = 4
      lmax["Mg"] = 4
      lmax["Al"] = 4
      lmax["Si"] = 4
      lmax["P"] = 4
      lmax["S"] = 4
      lmax["Cl"] = 4
      lmax["Ar"] = 4
      lmax["K"] = 4
      lmax["Ca"] = 4

      nmax["H", 0] = 2
      nmax["H", 1] = 2
      nmax["H", 2] = 2

      nmax["He", 0] = 4
      nmax["He", 1] = 3
      nmax["He", 2] = 2

      nmax["Li", 0] = 7
      nmax["Li", 1] = 5
      nmax["Li", 2] = 3
      nmax["Li", 3] = 2

      nmax["Be", 0] = 7
      nmax["Be", 1] = 5
      nmax["Be", 2] = 3
      nmax["Be", 3] = 2

      nmax["B", 0] = 10
      nmax["B", 1] = 8
      nmax["B", 2] = 5
      nmax["B", 3] = 1
      nmax["B", 4] = 1

      nmax["C", 0] = 10
      nmax["C", 1] = 8
      nmax["C", 2] = 5
      nmax["C", 3] = 1
      nmax["C", 4] = 1

      nmax["N", 0] = 10
      nmax["N", 1] = 8
      nmax["N", 2] = 4
      nmax["N", 3] = 2
      nmax["N", 4] = 1

      nmax["O", 0] = 10
      nmax["O", 1] = 8
      nmax["O", 2] = 4
      nmax["O", 3] = 2
      nmax["O", 4] = 1

      nmax["F", 0] = 10
      nmax["F", 1] = 8
      nmax["F", 2] = 4
      nmax["F", 3] = 2
      nmax["F", 4] = 1

      nmax["Ne", 0] = 10
      nmax["Ne", 1] = 8
      nmax["Ne", 2] = 4
      nmax["Ne", 3] = 2
      nmax["Ne", 4] = 1

      nmax["Na", 0] = 13
      nmax["Na", 1] = 9
      nmax["Na", 2] = 7
      nmax["Na", 3] = 4
      nmax["Na", 4] = 1

      nmax["Mg", 0] = 13
      nmax["Mg", 1] = 9
      nmax["Mg", 2] = 7
      nmax["Mg", 3] = 4
      nmax["Mg", 4] = 1

      nmax["Al", 0] = 13
      nmax["Al", 1] = 11
      nmax["Al", 2] = 9
      nmax["Al", 3] = 4
      nmax["Al", 4] = 1

      nmax["Si", 0] = 13
      nmax["Si", 1] = 11
      nmax["Si", 2] = 9
      nmax["Si", 3] = 4
      nmax["Si", 4] = 1

      nmax["P", 0] = 13
      nmax["P", 1] = 11
      nmax["P", 2] = 9
      nmax["P", 3] = 4
      nmax["P", 4] = 1

      nmax["S", 0] = 13
      nmax["S", 1] = 11
      nmax["S", 2] = 9
      nmax["S", 3] = 4
      nmax["S", 4] = 1

      nmax["Cl", 0] = 13
      nmax["Cl", 1] = 11
      nmax["Cl", 2] = 9
      nmax["Cl", 3] = 4
      nmax["Cl", 4] = 1

      nmax["Ar", 0] = 13
      nmax["Ar", 1] = 11
      nmax["Ar", 2] = 9
      nmax["Ar", 3] = 4
      nmax["Ar", 4] = 1

      nmax["K", 0] = 15
      nmax["K", 1] = 11
      nmax["K", 2] = 10
      nmax["K", 3] = 5
      nmax["K", 4] = 3

      nmax["Ca", 0] = 15
      nmax["Ca", 1] = 11
      nmax["Ca", 2] = 10
      nmax["Ca", 3] = 5
      nmax["Ca", 4] = 3

      return [lmax,nmax]


   if basis=="FHI-aims-clusters":
   
      lmax["H"] = 4
      lmax["C"] = 5
      lmax["N"] = 5
      lmax["O"] = 5
      lmax["F"] = 5

      # hydrogen
      nmax[("H",0)] = 9
      nmax[("H",1)] = 7
      nmax[("H",2)] = 6
      nmax[("H",3)] = 3
      nmax[("H",4)] = 1
      # oxygen
      nmax[("O",0)] = 9
      nmax[("O",1)] = 10
      nmax[("O",2)] = 9
      nmax[("O",3)] = 8
      nmax[("O",4)] = 6
      nmax[("O",5)] = 4
      # carbon
      nmax[("C",0)] = 11
      nmax[("C",1)] = 10
      nmax[("C",2)] = 9
      nmax[("C",3)] = 8
      nmax[("C",4)] = 7
      nmax[("C",5)] = 5
      # nitrogen
      nmax[("N",0)] = 9
      nmax[("N",1)] = 10
      nmax[("N",2)] = 9
      nmax[("N",3)] = 8
      nmax[("N",4)] = 6
      nmax[("N",5)] = 4
      # fluorine
      nmax[("F",0)] = 11
      nmax[("F",1)] = 10
      nmax[("F",2)] = 10
      nmax[("F",3)] = 7
      nmax[("F",4)] = 7
      nmax[("F",5)] = 5

      return [lmax,nmax]

   if basis=="RI-cc-pvqz":
      lmax["H"] = 4
      lmax["B"] = 5
      lmax["C"] = 5
      lmax["N"] = 5
      lmax["O"] = 5
      lmax["F"] = 5
      lmax["Al"] = 5
      lmax["Si"] = 5
      lmax["P"] = 5
      lmax["S"] = 5
      lmax["Cl"] = 5
      lmax["Ga"] = 5
      lmax["Ge"] = 5
      lmax["As"] = 5
      lmax["Se"] = 5
      lmax["Br"] = 5

      nmax["H", 0] = 4
      nmax["H", 1] = 3
      nmax["H", 2] = 3
      nmax["H", 3] = 2
      nmax["H", 4] = 1

      nmax["B", 0] = 10
      nmax["B", 1] = 7
      nmax["B", 2] = 5
      nmax["B", 3] = 3
      nmax["B", 4] = 2
      nmax["B", 5] = 1

      nmax["C", 0] = 10
      nmax["C", 1] = 7
      nmax["C", 2] = 5
      nmax["C", 3] = 3
      nmax["C", 4] = 2
      nmax["C", 5] = 1

      nmax["N", 0] = 10
      nmax["N", 1] = 7
      nmax["N", 2] = 5
      nmax["N", 3] = 3
      nmax["N", 4] = 2
      nmax["N", 5] = 1

      nmax["O", 0] = 10
      nmax["O", 1] = 7
      nmax["O", 2] = 5
      nmax["O", 3] = 3
      nmax["O", 4] = 2
      nmax["O", 5] = 1

      nmax["F", 0] = 10
      nmax["F", 1] = 7
      nmax["F", 2] = 5
      nmax["F", 3] = 3
      nmax["F", 4] = 2
      nmax["F", 5] = 1

      nmax["Al", 0] = 13
      nmax["Al", 1] = 11
      nmax["Al", 2] = 9
      nmax["Al", 3] = 4
      nmax["Al", 4] = 2
      nmax["Al", 5] = 1

      nmax["Si", 0] = 13
      nmax["Si", 1] = 11
      nmax["Si", 2] = 9
      nmax["Si", 3] = 4
      nmax["Si", 4] = 2
      nmax["Si", 5] = 1

      nmax["P", 0] = 13
      nmax["P", 1] = 11
      nmax["P", 2] = 9
      nmax["P", 3] = 4
      nmax["P", 4] = 2
      nmax["P", 5] = 1

      nmax["S", 0] = 13
      nmax["S", 1] = 11
      nmax["S", 2] = 9
      nmax["S", 3] = 4
      nmax["S", 4] = 2
      nmax["S", 5] = 1

      nmax["Cl", 0] = 13
      nmax["Cl", 1] = 11
      nmax["Cl", 2] = 9
      nmax["Cl", 3] = 4
      nmax["Cl", 4] = 2
      nmax["Cl", 5] = 1

      nmax["Ga", 0] = 14
      nmax["Ga", 1] = 13
      nmax["Ga", 2] = 11
      nmax["Ga", 3] = 9
      nmax["Ga", 4] = 4
      nmax["Ga", 5] = 1

      nmax["Ge", 0] = 14
      nmax["Ge", 1] = 13
      nmax["Ge", 2] = 11
      nmax["Ge", 3] = 9
      nmax["Ge", 4] = 4
      nmax["Ge", 5] = 1

      nmax["As", 0] = 14
      nmax["As", 1] = 13
      nmax["As", 2] = 11
      nmax["As", 3] = 9
      nmax["As", 4] = 4
      nmax["As", 5] = 1

      nmax["Se", 0] = 14
      nmax["Se", 1] = 13
      nmax["Se", 2] = 11
      nmax["Se", 3] = 9
      nmax["Se", 4] = 4
      nmax["Se", 5] = 1

      nmax["Br", 0] = 14
      nmax["Br", 1] = 13
      nmax["Br", 2] = 11
      nmax["Br", 3] = 9
      nmax["Br", 4] = 4
      nmax["Br", 5] = 1

      return [lmax,nmax]
   
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

      return [lmax,nmax]
      
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

      return [lmax,nmax]

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

   if basis=="LRI-DZVP-MOLOPT-GTH-MEDIUM":
      
      lmax["H"] = 3
      lmax["O"] = 4
      lmax["Cu"] = 6

      # H
      nmax[("H",0)] = 10
      nmax[("H",1)] = 9 
      nmax[("H",2)] = 8 
      nmax[("H",3)] = 6
      # O
      nmax[("O",0)] = 15
      nmax[("O",1)] = 13
      nmax[("O",2)] = 12
      nmax[("O",3)] = 11
      nmax[("O",4)] = 9
      # Cu    
      nmax[("Cu",0)] = 15 
      nmax[("Cu",1)] = 13
      nmax[("Cu",2)] = 12
      nmax[("Cu",3)] = 11
      nmax[("Cu",4)] = 10
      nmax[("Cu",5)] = 9
      nmax[("Cu",6)] = 8

      return [lmax,nmax]

   if basis=="LRI-DZVP-MOLOPT-GTH-MEDIUM-FULL-ANGULAR":

      lmax["H"] = 3
      lmax["O"] = 4
      lmax["Cu"] = 6

      # H
      nmax[("H",0)] = 10
      nmax[("H",1)] = 10
      nmax[("H",2)] = 10
      nmax[("H",3)] = 10
      # O
      nmax[("O",0)] = 15
      nmax[("O",1)] = 15
      nmax[("O",2)] = 15
      nmax[("O",3)] = 15
      nmax[("O",4)] = 15
      # Cu    
      nmax[("Cu",0)] = 15
      nmax[("Cu",1)] = 15
      nmax[("Cu",2)] = 15
      nmax[("Cu",3)] = 15
      nmax[("Cu",4)] = 15
      nmax[("Cu",5)] = 15
      nmax[("Cu",6)] = 15

      return [lmax,nmax]

   if basis=="LRI-DZVP-MOLOPT-GTH-MEDIUM-FULL-ANGULAR-SUPER-FAT-WIDER-18":

      lmax["Cu"] = 6

      # Cu    
      nmax[("Cu",0)] = 18
      nmax[("Cu",1)] = 18
      nmax[("Cu",2)] = 18
      nmax[("Cu",3)] = 18
      nmax[("Cu",4)] = 18
      nmax[("Cu",5)] = 18
      nmax[("Cu",6)] = 18

      return [lmax,nmax]

   if basis=="DF-DZVP-MOLOPT-GTH":

      lmax["Ag"] = 6 

      nmax[("Ag",0)] = 15
      nmax[("Ag",1)] = 15
      nmax[("Ag",2)] = 15
      nmax[("Ag",3)] = 15
      nmax[("Ag",4)] = 15
      nmax[("Ag",5)] = 15
      nmax[("Ag",6)] = 15

      lmax["Au"] = 6 

      nmax[("Au",0)] = 18
      nmax[("Au",1)] = 18
      nmax[("Au",2)] = 18
      nmax[("Au",3)] = 18
      nmax[("Au",4)] = 18
      nmax[("Au",5)] = 18
      nmax[("Au",6)] = 18

      return [lmax,nmax]

   if basis=="FHI-aims-tight":

      lmax["H"] = 4
      lmax["O"] = 8

      nmax[("H",0)] = 9
      nmax[("H",1)] = 7
      nmax[("H",2)] = 6
      nmax[("H",3)] = 3
      nmax[("H",4)] = 1

      nmax[("O",0)] = 9
      nmax[("O",1)] = 10
      nmax[("O",2)] = 9
      nmax[("O",3)] = 8
      nmax[("O",4)] = 6
      nmax[("O",5)] = 4
      nmax[("O",6)] = 4
      nmax[("O",7)] = 2
      nmax[("O",8)] = 1

      return [lmax,nmax]

   if basis=="FHI-aims-ZrS":

      lmax["S"] = 5
      lmax["Zr"] = 5

      nmax[("S",0)] = 10
      nmax[("S",1)] = 10
      nmax[("S",2)] = 10
      nmax[("S",3)] = 10
      nmax[("S",4)] = 8
      nmax[("S",5)] = 5

      nmax[("Zr",0)] = 14
      nmax[("Zr",1)] = 13
      nmax[("Zr",2)] = 13
      nmax[("Zr",3)] = 12
      nmax[("Zr",4)] = 12
      nmax[("Zr",5)] = 8

      return [lmax,nmax]

   if basis=="FHI-aims-graphene":

      lmax["C"] = 4

      nmax[("C",0)] = 7
      nmax[("C",1)] = 7
      nmax[("C",2)] = 6
      nmax[("C",3)] = 5
      nmax[("C",4)] = 3

      return [lmax,nmax]

   if basis=="FHI-aims-MoSe":

      lmax["Se"] = 6
      lmax["Mo"] = 6

      nmax[("Se",0)] = 13
      nmax[("Se",1)] = 13
      nmax[("Se",2)] = 11
      nmax[("Se",3)] = 11
      nmax[("Se",4)] = 9
      nmax[("Se",5)] = 7
      nmax[("Se",6)] = 4

      nmax[("Mo",0)] = 14
      nmax[("Mo",1)] = 13
      nmax[("Mo",2)] = 13
      nmax[("Mo",3)] = 12
      nmax[("Mo",4)] = 11
      nmax[("Mo",5)] = 8
      nmax[("Mo",6)] = 5

      return [lmax,nmax]

   if basis=="FHI-aims-light":

      lmax["H"] = 2
      lmax["O"] = 4
      lmax["C"] = 4

      nmax[("H",0)] = 4
      nmax[("H",1)] = 3
      nmax[("H",2)] = 1

      nmax[("O",0)] = 9
      nmax[("O",1)] = 9
      nmax[("O",2)] = 8
      nmax[("O",3)] = 3
      nmax[("O",4)] = 1

      nmax[("C",0)] = 9
      nmax[("C",1)] = 8
      nmax[("C",2)] = 7
      nmax[("C",3)] = 3
      nmax[("C",4)] = 1

      return [lmax,nmax]

   if basis=="RI_AUTO_OPT-ccGRB":

      lmax["Au"] = 6

      nmax[("Au",0)] = 8
      nmax[("Au",1)] = 8
      nmax[("Au",2)] = 8
      nmax[("Au",3)] = 8
      nmax[("Au",4)] = 8
      nmax[("Au",5)] = 1
      nmax[("Au",6)] = 1

      return [lmax,nmax]

