SUBROUTINE prediction(kernsizes,specarray,atomcount,atomicindx,nspecies,ntest,natmax,&
                      llmax,nnmax,natoms,testrange,test_species,almax,anmax,M,ww,coeffs)

use omp_lib
IMPLICIT NONE
! allocate I/O variables
INTEGER:: ntest,natmax,llmax,nnmax,M,nspecies
INTEGER,DIMENSION(ntest)::natoms,testrange,kernsizes 
INTEGER,DIMENSION(ntest,natmax)::test_species 
INTEGER,DIMENSION(nspecies)::almax
INTEGER,DIMENSION(nspecies,llmax+1)::anmax
INTEGER,DIMENSION(M)::specarray
INTEGER,DIMENSION(ntest,nspecies)::atomcount
INTEGER,DIMENSION(natmax,nspecies,ntest)::atomicindx
REAL*8,DIMENSION(M,llmax+1,nnmax,2*llmax+1):: ww 
REAL*8,DIMENSION(ntest,natmax,llmax+1,nnmax,2*llmax+1):: coeffs 
! allocate internal variables
INTEGER,DIMENSION(:,:,:,:,:), ALLOCATABLE :: kernsparseindexes
REAL*8,DIMENSION(:), ALLOCATABLE :: kernels
REAL*8:: kern
INTEGER:: itest,iat,ispe,al,l,msize,n,im,iref,imm,conf,ik,a1,sk,ifk,icspe
CHARACTER*16:: conf_str,ref_str

!f2py intent(in) kernsizes,specarray,atomcount,atomicindx,nspecies,ntest
!f2py intent(in) natmax,llmax,nnmax,natoms,testrange,test_species,almax,anmax,M,ww
!f2py intent(out) coeffs 
!f2py depend(ntest) coeffs,natoms,test_species,testrange,atomcount,atomicindx,kernsizes 
!f2py depend(M) ww,specarray
!f2py depend(llmax) coeffs,anmax,ww
!f2py depend(natmax) coeffs,test_species,atomicindx
!f2py depend(nnmax) coeffs,ww 
!f2py depend(nspecies) almax,anmax,atomcount,atomicindx 

coeffs(:,:,:,:,:) = 0.d0 

!$OMP PARALLEL DEFAULT(private) &
!$OMP FIRSTPRIVATE(specarray,atomcount,atomicindx,nspecies,ntest,natmax,llmax,nnmax,natoms) &
!$OMP FIRSTPRIVATE(testrange,test_species,almax,anmax,M,kernsparseindexes,kernels,kernsizes) &
!$OMP SHARED(coeffs,ww)
!$OMP DO SCHEDULE(dynamic)
do itest=1,ntest
   allocate(kernels(kernsizes(itest)))
   kernels(:) = 0.d0
   allocate(kernsparseindexes(2*llmax+1,natoms(itest),2*llmax+1,llmax+1,M))
   kernsparseindexes(:,:,:,:,:) = 0
   conf = testrange(itest)
   write(conf_str,*) conf
   write(ref_str,*) M
   ifk = itest + 1000
   open(unit=ifk,file='kernels/kernel_conf'//trim(adjustl(conf_str))//'.dat',action='read',status='old')
   ik = 1
   do iref=1,M
      ispe = specarray(iref) + 1
      do l=1,almax(ispe)
         msize = 2*(l-1)+1
         do im=1,msize
            do icspe=1,atomcount(itest,ispe)
               do imm=1,msize
                  read(ifk,*) kern
                  kernels(ik) = kern
                  kernsparseindexes(imm,icspe,im,l,iref) = ik
                  ik = ik + 1
               enddo
            enddo
         enddo
      enddo
   enddo
   close(ifk)
   do iref=1,M
      ispe = specarray(iref) + 1
      do l=1,almax(ispe)
         msize = 2*(l-1)+1
         do n=1,anmax(ispe,l)
            do im=1,msize
               do icspe=1,atomcount(itest,ispe)
                  iat = atomicindx(icspe,ispe,itest) + 1
                  do imm=1,msize
                     sk = kernsparseindexes(imm,icspe,im,l,iref)
                     coeffs(itest,iat,l,n,imm) = coeffs(itest,iat,l,n,imm) + kernels(sk) * ww(iref,l,n,im)
                  enddo
               enddo
            enddo
         enddo
      enddo
   enddo
   deallocate(kernsparseindexes)
   deallocate(kernels)
enddo
!$OMP END DO
!$OMP END PARALLEL

return
END 
