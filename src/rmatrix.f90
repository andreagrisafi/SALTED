SUBROUTINE rmatrix(llmax,nspecies,M,totsize,specarray,almax,ancut,kMM,Rmat)

use omp_lib
IMPLICIT NONE
! allocate I/O variables
INTEGER:: M,totsize,llmax,nspecies
INTEGER,DIMENSION(M):: specarray
INTEGER,DIMENSION(nspecies):: almax
INTEGER,DIMENSION(nspecies,llmax+1):: ancut
REAL*8,DIMENSION(llmax+1,M*(2*llmax+1),M*(2*llmax+1)):: kMM
REAL*8,DIMENSION(totsize,totsize):: Rmat
! allocate internal variables
INTEGER:: iat,jat,l1,l2,n1,n2,im1,im2,i1,i2,a1,a2,al1,al2,anc1,anc2,conf,if1,if2
INTEGER:: iref1,iref2,msize1,msize2,ik1,ik2
CHARACTER*32:: conf_str 

!f2py intent(in) llmax,nspecies,M,totsize,specarray,almax,ancut,kMM
!f2py intent(out) Rmat 
!f2py depend(totsize) Rmat 
!f2py depend(nspecies) almax,ancut
!f2py depend(M) specarray,kMM 
!f2py depend(llmax) ancut,kMM

Rmat(:,:) = 0.d0

i1 = 1
do iref1=1,M
   a1 = specarray(iref1) + 1
   al1 = almax(a1)
   do l1=1,al1
      msize1 = 2*(l1-1)+1
      anc1 = ancut(a1,l1)
      do n1=1,anc1
         do im1=1,msize1
            ik1 = (iref1-1)*msize1+im1
            ! Loop over 2nd dimension
            i2 = 1
            do iref2=1,M
               a2 = specarray(iref2) + 1 
               al2 = almax(a2)
               do l2=1,al2
                  msize2 = 2*(l2-1)+1
                  anc2 = ancut(a2,l2)
                  do n2=1,anc2
                     do im2=1,msize2
                        ik2 = (iref2-1)*msize2+im2
                        if (a1==a2.and.l1==l2.and.n1==n2) then
                           Rmat(i1,i2) = kMM(l1,ik1,ik2) 
                        endif
                        i2 = i2 + 1
                     enddo
                  enddo
               enddo
            enddo
            i1 = i1 + 1
         enddo
      enddo
   enddo
enddo

return
END 
