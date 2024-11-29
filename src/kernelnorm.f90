SUBROUTINE kernelnorm(n1,n2,msize,normfact1,normfact2,kernel,knorm)

!use omp_lib
IMPLICIT NONE
INTEGER:: n1,n2,msize,i1,i2,j1,j2,im1,im2 
REAL*8, DIMENSION(n2*msize,n1*msize):: kernel, knorm 
REAL*8, DIMENSION(n1):: normfact1 
REAL*8, DIMENSION(n2):: normfact2

!f2py intent(in) n1,n2,msize,normfact1,normfact2,kernel
!f2py intent(out) knorm
!f2py depend(n1) kernel, knorm , normfact1
!f2py depend(n2) kernel, knorm , normfact2
!f2py depend(msize) kernel, knorm

j1 = 1
do i1=1,n1
   do im1=1,msize
      j2 = 1
      do i2=1,n2
         do im2=1,msize
            knorm(j2,j1) = kernel(j2,j1) / dsqrt(normfact1(i1)*normfact2(i2))
            j2 = j2 + 1
         enddo
      enddo
      j1 = j1 + 1
   enddo
enddo

return
END
