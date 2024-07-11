SUBROUTINE kernelequicomb(n1,n2,lam1,lam2,L,nsize,msize,&
                         cgsize,cgcoefs,knm,k0,kernel)

!use omp_lib
IMPLICIT NONE
INTEGER:: n1,n2,lam1,lam2,L,nsize,msize,cgsize
REAL*8, DIMENSION(cgsize):: cgcoefs
REAL*8, DIMENSION(n2,n1):: k0 
COMPLEX*16, DIMENSION(n2*(2*L+1),n1*(2*L+1)):: knm
COMPLEX*16, DIMENSION(msize,nsize):: kernel 

INTEGER:: i1,i2,iM1,iM2,idx1,idx2,icg1,icg2,mu1size,mu2size
INTEGER:: imu1,imu2,mu1,mu2,ik1,ik2,k1,k2,M1,M2,j1,j2
REAL*8:: cg1,cg2

!f2py intent(in) n1,n2,lam1,lam2,L,nsize,msize
!f2py intent(in) cgsize,cgcoefs,knm,k0
!f2py intent(out) kernel
!f2py depend(nsize) kernel 
!f2py depend(msize) kernel 
!f2py depend(cgsize) cgcoefs
!f2py depend(n1) k0,knm 
!f2py depend(n2) k0,knm
!f2py depend(L) knm

mu1size = 2*lam1+1
mu2size = 2*lam2+1

kernel = 0.d0

iM1 = 1
idx1 = 1
do i1=1,n1
   icg1 = 1
   do imu1=1,mu1size
      mu1 = imu1-1-lam1 
      do ik1=1,mu2size
         k1 = ik1-1-lam2
         M1 = mu1+k1
         if (abs(M1).le.L) then
            j1 = M1+L
            cg1 = cgcoefs(icg1)
            iM2 = 1
            idx2 = 1
            do i2=1,n2
               icg2 = 1
               do imu2=1,mu1size
                  mu2 = imu2-1-lam1 
                  do ik2=1,mu2size
                     k2 = ik2-1-lam2
                     M2 = mu2+k2
                     if (abs(M2).le.L) then 
                        j2 = M2+L
                        cg2 = cgcoefs(icg2)
                        kernel(iM2,iM1) = kernel(iM2,iM1) + cg1 * cg2 & 
                            * knm(idx2+j2,idx1+j1) * k0(i2,i1)
                        icg2 = icg2 + 1
                     endif
                     iM2 = iM2 + 1
                  enddo
               enddo
               idx2 = idx2 + 2*L+1
            enddo
            icg1 = icg1 + 1
         endif
         iM1 = iM1 + 1
      enddo
   enddo
   idx1 = idx1 + 2*L+1
enddo

return
END
