SUBROUTINE equicombsparse_grad(natoms,nang1,nang2,nrad1,nrad2,v1,v2,dv1,dv2,&
                    wigdim,w3j,llmax,llvec,lam,c2r,&
                    featsize,nfps,vfps,p,grad_p)

!use omp_lib
IMPLICIT NONE
INTEGER:: natoms,nang1,nang2,nrad1,nrad2,llmax,lam,wigdim,ifps,ifeat,n
INTEGER:: iat,i_grad,n1,n2,iwig,l1,l2,il,imu,im1,im2,mu,m1,m2,featsize,nfps
INTEGER, DIMENSION(nfps):: vfps
INTEGER, DIMENSION(2,llmax):: llvec
REAL*8, DIMENSION(wigdim):: w3j
REAL*8, DIMENSION(2*lam+1):: preal
REAL*8, ALLOCATABLE :: grad_preal(:,:,:)
COMPLEX*16, DIMENSION(2*lam+1):: pcmplx
COMPLEX*16, ALLOCATABLE :: grad_pcmplx(:,:,:)
COMPLEX*16, DIMENSION(2*lam+1,2*lam+1):: c2r
COMPLEX*16, DIMENSION(2*nang1+1,nang1+1,nrad1,natoms):: v1
COMPLEX*16, DIMENSION(2*nang2+1,nang2+1,nrad2,natoms):: v2
COMPLEX*16, DIMENSION(natoms,3,2*nang1+1,nang1+1,nrad1,natoms):: dv1
COMPLEX*16, DIMENSION(natoms,3,2*nang2+1,nang2+1,nrad2,natoms):: dv2
COMPLEX*16:: v1v, v2c, dv2c1, dv2c2, dv2c3
REAL*8, ALLOCATABLE:: ptemp(:,:)
REAL*8, ALLOCATABLE :: grad_ptemp (:,:,:,:)
REAL*8, DIMENSION(2*lam+1,nfps,natoms):: p
REAL*8, DIMENSION(natoms,3,2*lam+1,nfps,natoms):: grad_p
REAL*8, ALLOCATABLE :: dot1(:), dot2(:), dot3(:)
REAL*8:: inner, normfact, normfact3

!f2py intent(in) natoms,nang1,nang2,nrad1,nrad2,v1,v2,dv1,dv2,wigdim,w3j,llmax,llvec,lam,c2r
!f2py intent(in) featsize
!f2py intent(out) p, grad_p
!f2py depend(natoms) p, grad_p, v1, v2, dv1, dv2
!f2py depend(nrad1) v1, dv1
!f2py depend(nrad2) v2, dv2
!f2py depend(nang1) v1, dv1
!f2py depend(nang2) v2, dv2
!f2py depend(lam) p, grad_p, c2r
!f2py depend(llmax) llvec
!f2py depend(wigdim) w3j 
!f2py depend(nfps) vfps, p, grad_p

p = 0.d0
grad_p = 0.d0
inner = 0.d0
preal = 0.d0

!$OMP PARALLEL DEFAULT(NONE) &
!$OMP& SHARED(p,grad_p,v1,v2,dv1,dv2,w3j,natoms,nang1,nang2, &
!$OMP&        nrad1,nrad2,llmax,lam,wigdim,llvec,c2r,featsize, &
!$OMP&        nfps,vfps) &
!$OMP& PRIVATE(n,iat,i_grad,n1,n2,iwig,l1,l2,il,imu,im1,im2,mu, &
!$OMP&         m1,m2,ifeat,preal,grad_preal,pcmplx,grad_pcmplx, &
!$OMP&         v2c,dv2c1,dv2c2,dv2c3,ptemp,grad_ptemp,inner, &
!$OMP&         normfact,dot1,dot2,dot3,ifps,normfact3,v1v)
allocate(grad_preal(natoms,3,2*lam+1))
allocate(grad_pcmplx(natoms,3,2*lam+1))
allocate(grad_ptemp(natoms,3,2*lam+1,featsize))
allocate(ptemp(2*lam+1,featsize))
allocate(dot1(natoms))
allocate(dot2(natoms))
allocate(dot3(natoms))
!$OMP DO SCHEDULE(dynamic)
do iat=1,natoms
   inner = 0.d0
   dot1 = 0.d0
   dot2 = 0.d0
   dot3 = 0.d0
   ptemp = 0.d0
   grad_ptemp = 0.d0
   preal = 0.d0
   grad_preal = 0.d0
   ifeat = 1
   do n1=1,nrad1
      do n2=1,nrad2
         iwig = 1
         do il=1,llmax
            l1 = llvec(1,il)
            l2 = llvec(2,il)
            pcmplx = dcmplx(0.d0,0.d0)
            grad_pcmplx = dcmplx(0.d0,0.d0)
            do imu=1,2*lam+1
               mu = imu-1-lam
               do im1=1,2*l1+1
                  m1 = im1-1-l1
                  m2 = m1-mu
                  if (abs(m2)<=l2) then
                     im2 = m2+l2+1
                     v2c  = dconjg(v2(im2,l2+1,n2,iat))
                     v1v = v1(im1,l1+1,n1,iat)
                     pcmplx(imu) = pcmplx(imu) &
                                 + w3j(iwig) * v1v * v2c
                     do i_grad=1,natoms
                        dv2c1 = dconjg(dv2(i_grad,1,im2,l2+1,n2,iat))
                        dv2c2 = dconjg(dv2(i_grad,2,im2,l2+1,n2,iat))
                        dv2c3 = dconjg(dv2(i_grad,3,im2,l2+1,n2,iat))
                        grad_pcmplx(i_grad,1,imu) = grad_pcmplx(i_grad,1,imu) &
                                 + w3j(iwig) * dv1(i_grad,1,im1,l1+1,n1,iat) * v2c&
                                 + w3j(iwig) * v1v * dv2c1
                        grad_pcmplx(i_grad,2,imu) = grad_pcmplx(i_grad,2,imu) &
                                 + w3j(iwig) * dv1(i_grad,2,im1,l1+1,n1,iat) * v2c&
                                 + w3j(iwig) * v1v * dv2c2
                        grad_pcmplx(i_grad,3,imu) = grad_pcmplx(i_grad,3,imu) &
                                 + w3j(iwig) * dv1(i_grad,3,im1,l1+1,n1,iat) * v2c&
                                 + w3j(iwig) * v1v * dv2c3
                     enddo
                     iwig = iwig + 1
                  endif
               enddo
            enddo
            preal = 0.d0
            grad_preal = 0.d0
            do imu = 1, 2*lam+1
                do im1 = 1, 2*lam+1
                    preal(imu) = preal(imu) + dreal(c2r(imu,im1) * pcmplx(im1))
                    do i_grad = 1, natoms
                        grad_preal(i_grad,1,imu) = grad_preal(i_grad,1,imu) + &
                        dreal(c2r(imu,im1) * grad_pcmplx(i_grad,1,im1))
                        grad_preal(i_grad,2,imu) = grad_preal(i_grad,2,imu) + &
                        dreal(c2r(imu,im1) * grad_pcmplx(i_grad,2,im1))
                        grad_preal(i_grad,3,imu) = grad_preal(i_grad,3,imu) + &
                        dreal(c2r(imu,im1) * grad_pcmplx(i_grad,3,im1))
                    enddo
                enddo
            enddo
            do imu=1,2*lam+1
               inner = inner + preal(imu)**2
               ptemp(imu,ifeat) = preal(imu)
               do i_grad=1,natoms
                  dot1(i_grad) = dot1(i_grad) + preal(imu)*grad_preal(i_grad,1,imu)
                  dot2(i_grad) = dot2(i_grad) + preal(imu)*grad_preal(i_grad,2,imu)
                  dot3(i_grad) = dot3(i_grad) + preal(imu)*grad_preal(i_grad,3,imu)
                  grad_ptemp(i_grad,1,imu,ifeat) = grad_preal(i_grad,1,imu)
                  grad_ptemp(i_grad,2,imu,ifeat) = grad_preal(i_grad,2,imu)
                  grad_ptemp(i_grad,3,imu,ifeat) = grad_preal(i_grad,3,imu)
               enddo
            enddo
            ifeat = ifeat + 1
         enddo
      enddo
   enddo
   normfact = dsqrt(inner)
   normfact3 = normfact**3
   !if (inner < tiny(1.0d0)) cycle
   do n=1,nfps
      ifps = vfps(n) + 1
      do i_grad=1,natoms
         !dot1 = dot_product(ptemp(:,ifps), grad_ptemp(1,i_grad,:,ifps))
         !dot2 = dot_product(ptemp(:,:), grad_ptemp(2,i_grad,:,:))
         !dot3 = dot_product(ptemp(:,:), grad_ptemp(3,i_grad,:,:))
         do imu=1,2*lam+1
            p(imu,n,iat) = ptemp(imu,ifps) / normfact
            grad_p(i_grad,1,imu,n,iat) = (grad_ptemp(i_grad,1,imu,ifps) / normfact) &
                    - ptemp(imu,ifps) * dot1(i_grad) / (normfact3)
            grad_p(i_grad,2,imu,n,iat) = (grad_ptemp(i_grad,2,imu,ifps) / normfact) &
                    - ptemp(imu,ifps) * dot2(i_grad) / (normfact3)
            grad_p(i_grad,3,imu,n,iat) = (grad_ptemp(i_grad,3,imu,ifps) / normfact) &
                    - ptemp(imu,ifps) * dot3(i_grad) / (normfact3)
         enddo
      enddo
   enddo
enddo
!$OMP END DO
deallocate(grad_preal,grad_pcmplx,grad_ptemp,ptemp,dot1,dot2,dot3)
!$OMP END PARALLEL

return
END

