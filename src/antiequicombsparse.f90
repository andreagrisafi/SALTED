SUBROUTINE antiequicombsparse(natoms,nang1,nang2,nrad1,nrad2,v1,v2,&
                              wigdim,w3j,llmax,llvec,lam,c2r,&
                              featsize,nfps,vfps,p)

!use omp_lib
IMPLICIT NONE
INTEGER:: natoms,nang1,nang2,nrad1,nrad2,llmax,lam,wigdim,ifps,ifeat,n
INTEGER:: iat,n1,n2,iwig,l1,l2,il,imu,im1,im2,mu,m1,m2,featsize,nfps
INTEGER, DIMENSION(nfps):: vfps 
INTEGER, DIMENSION(2,llmax):: llvec
REAL*8, DIMENSION(wigdim):: w3j
REAL*8, DIMENSION(2*lam+1):: pimag
COMPLEX*16, DIMENSION(2*lam+1):: pcmplx 
COMPLEX*16, DIMENSION(2*lam+1,2*lam+1):: c2r
COMPLEX*16, DIMENSION(2*nang1+1,nang1+1,nrad1,natoms):: v1 
COMPLEX*16, DIMENSION(2*nang2+1,nang2+1,nrad2,natoms):: v2 
REAL*8, DIMENSION(2*lam+1,featsize):: ptemp 
REAL*8, DIMENSION(2*lam+1,nfps,natoms):: p 
REAL*8:: inner,normfact

!f2py intent(in) natoms,nang1,nang2,nrad1,nrad2,v1,v2,wigdim,w3j,llmax,llvec,lam,c2r
!f2py intent(in) featsize, nfps, vfps  
!f2py intent(out) p 
!f2py depend(natoms) p, v1, v2
!f2py depend(nrad1) v1 
!f2py depend(nrad2) v2
!f2py depend(nang1) v1 
!f2py depend(nang2) v2
!f2py depend(lam) p, c2r
!f2py depend(llmax) llvec
!f2py depend(wigdim) w3j 
!f2py depend(nfps) vfps, p 

p = 0.d0

!$OMP PARALLEL DEFAULT(private) &
!$OMP FIRSTPRIVATE(natoms,nang1,nang2,nrad1,nrad2,w3j,llmax,llvec,lam,c2r,nfps,vfps) &
!$OMP SHARED(p,v1,v2)
!$OMP DO SCHEDULE(dynamic)
do iat=1,natoms
   inner = 0.0
   ptemp = 0.0
   ifeat = 1
   do n1=1,nrad1
      do n2=1,nrad2 
         iwig = 1
         do il=1,llmax
            l1 = llvec(1,il)
            l2 = llvec(2,il)
            pcmplx = dcmplx(0.0,0.0)
            do imu=1,2*lam+1 
               mu = imu-1-lam
               do im1=1,2*l1+1
                  m1 = im1-1-l1
                  m2 = m1-mu
                  if (abs(m2)<=l2) then 
                     im2 = m2+l2+1
                     pcmplx(imu) = pcmplx(imu) &
                                 + w3j(iwig) * v1(im1,l1+1,n1,iat) * dconjg(v2(im2,l2+1,n2,iat))
                     iwig = iwig + 1
                  endif
               enddo
            enddo
            pimag = dimag(matmul(c2r,pcmplx))
            do imu=1,2*lam+1 
               inner = inner + pimag(imu)**2
               ptemp(imu,ifeat) = pimag(imu)
            enddo
            ifeat = ifeat + 1
         enddo
      enddo
   enddo
   normfact = dsqrt(inner)
   do n=1,nfps
      ifps = vfps(n) + 1
      do imu=1,2*lam+1 
         p(imu,n,iat) = ptemp(imu,ifps) / normfact 
      enddo
   enddo
enddo
!$OMP END DO
!$OMP END PARALLEL

return
END
