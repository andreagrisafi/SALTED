SUBROUTINE equicombfps(natoms, nang1, nang2, nrad1, nrad2, v1, v2, &
                    wigdim, w3j, llmax, llvec, lam, c2r, featsize, pvec)

use omp_lib
IMPLICIT NONE
INTEGER :: natoms, nang1, nang2, nrad1, nrad2, llmax, lam, wigdim, ifeat
INTEGER :: iat, n1, n2, iwig, l1, l2, il, imu, im1, im2, mu, m1, m2, featsize, temp1, temp2
INTEGER, DIMENSION(2, llmax) :: llvec
REAL*8, DIMENSION(wigdim) :: w3j
REAL*8, DIMENSION(2*lam+1) :: preal
COMPLEX*16, DIMENSION(2*lam+1) :: pcmplx 
COMPLEX*16, DIMENSION(2*lam+1, 2*lam+1) :: c2r
COMPLEX*16, DIMENSION(2*nang1+1, nang1+1, nrad1, natoms) :: v1 
COMPLEX*16, DIMENSION(2*nang2+1, nang2+1, nrad2, natoms) :: v2 
REAL*8, DIMENSION(2*lam+1, featsize) :: ptemp 
REAL*8, DIMENSION(featsize, natoms * (2*lam+1)) :: pvec
REAL*8 :: inner, normfact

!f2py intent(in) natoms, nang1, nang2, nrad1, nrad2, v1, v2, wigdim, w3j, llmax, llvec, lam, c2r
!f2py intent(in) featsize
!f2py intent(out) pvec
!f2py depend(natoms) pvec, v1, v2
!f2py depend(nrad1) v1 
!f2py depend(nrad2) v2
!f2py depend(nang1) v1 
!f2py depend(nang2) v2
!f2py depend(lam) pvec, c2r
!f2py depend(llmax) llvec
!f2py depend(wigdim) w3j 
!f2py depend(featsize) pvec

!$OMP PARALLEL DEFAULT(private) &
!$OMP FIRSTPRIVATE(natoms, nang1, nang2, nrad1, nrad2, w3j, llmax, llvec, lam, c2r, featsize) &
!$OMP SHARED(pvec, v1, v2)
!$OMP DO SCHEDULE(dynamic)
do iat = 1, natoms
   inner = 0.0
   ptemp = 0.0
   ifeat = 1
   do n1 = 1, nrad1
      do n2 = 1, nrad2 
         iwig = 1
         do il = 1, llmax
            l1 = llvec(1, il)
            l2 = llvec(2, il)
            pcmplx = dcmplx(0.0, 0.0)
            do imu = 1, 2*lam+1 
               mu = imu - 1 - lam
               do im1 = 1, 2*l1+1
                  m1 = im1 - 1 - l1
                  m2 = m1 - mu
                  if (abs(m2) <= l2) then 
                     im2 = m2 + l2 + 1
                     pcmplx(imu) = pcmplx(imu) &
                                 + w3j(iwig) * v1(im1, l1+1, n1, iat) * dconjg(v2(im2, l2+1, n2, iat))
                     iwig = iwig + 1
                  endif
               enddo
            enddo
            preal = dreal(matmul(c2r, pcmplx))
            do imu = 1, 2*lam+1 
               inner = inner + preal(imu)**2
               ptemp(imu, ifeat) = preal(imu)
            enddo
            ifeat = ifeat + 1
         enddo
      enddo
   enddo
   normfact = dsqrt(inner)
   do ifeat = 1, featsize
      do imu = 1, 2*lam+1 
         pvec(ifeat, (iat-1)*(2*lam+1) + imu) = ptemp(imu, ifeat) / normfact
      enddo
   enddo
enddo
!$OMP END DO
!$OMP END PARALLEL

return
END SUBROUTINE equicombfps