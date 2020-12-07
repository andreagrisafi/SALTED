SUBROUTINE getab(dirker,dirover,dirprojs,&
           trrange,atomspe,llmax,nnmax,nspecies,ntrain,M,natmax,natoms,totsize,&
           atomicindx,atomcount,specarray,almax,ancut,totalsizes,kernsizes,Avec,Bmat)
use omp_lib
IMPLICIT NONE
! allocate I/O variables
CHARACTER*32:: dirker,dirover,dirprojs 
INTEGER:: ntrain,M,natmax,totsize,llmax,nnmax,nspecies
INTEGER,DIMENSION(natmax,nspecies,ntrain):: atomicindx
INTEGER,DIMENSION(ntrain,nspecies):: atomcount
INTEGER,DIMENSION(ntrain):: trrange,natoms,totalsizes,kernsizes 
INTEGER,DIMENSION(ntrain,natmax):: atomspe 
INTEGER,DIMENSION(M):: specarray
INTEGER,DIMENSION(nspecies):: almax
INTEGER,DIMENSION(nspecies,llmax+1):: ancut
REAL*8,DIMENSION(totsize):: Avec
REAL*8,DIMENSION(totsize,totsize):: Bmat
! allocate internal variables
INTEGER:: iat,jat,l1,l2,n1,n2,im1,im2,i1,i2,a1,a2,al1,al2,anc1,anc2,conf,if1,if2,if3,ik1,ik2
INTEGER:: itrain,iref1,iref2,imm1,imm2,msize1,msize2,icspe1,icspe2,k1,k2,it1,it2,sp1,sp2,sk1,sk2
REAL*8:: Btemp,contrA,contrB,proj,over,kern
CHARACTER*16:: conf_str,ref_str
! allocatable arrays
INTEGER,DIMENSION(:,:,:,:,:), ALLOCATABLE :: kernsparseindexes 
INTEGER,DIMENSION(:,:,:,:), ALLOCATABLE :: sparseindexes 
REAL*8,DIMENSION(:), ALLOCATABLE :: projections
REAL*8,DIMENSION(:,:), ALLOCATABLE :: overlaps
REAL*8,DIMENSION(:), ALLOCATABLE :: kernels 

!f2py intent(in) dirker,dirover,dirprojs,trrange,atomspe,llmax,nnmax,nspecies,ntrain,M,natmax,natoms
!f2py intent(in) totsize,atomicindx,atomcount,specarray,almax,ancut,totalsizes,kernsizes
!f2py intent(out) Avec,Bmat 
!f2py depend(totsize) Avec,Bmat 
!f2py depend(nspecies) atomcount,atomicindx,almax,ancut
!f2py depend(M) specarray
!f2py depend(llmax) ancut
!f2py depend(natmax) atomicindx,atomspe
!f2py depend(ntrain) trrange,natoms,atomspe,atomcount,atomicindx,totalsizes,kernsizes

!$OMP PARALLEL DEFAULT(private) &
!$OMP FIRSTPRIVATE(kernels,projections,overlaps,trrange,atomspe,llmax,nnmax,nspecies,ntrain,natoms) &
!$OMP FIRSTPRIVATE(M,natmax,totsize,atomicindx,atomcount,specarray,almax,ancut) &
!$OMP FIRSTPRIVATE(totalsizes,sparseindexes,kernsparseindexes,kernsizes) &
!$OMP SHARED(Avec,Bmat,dirker,dirover,dirprojs)
!$OMP DO SCHEDULE(dynamic)
do itrain=1,ntrain
   allocate(projections(totalsizes(itrain)))
   projections(:) = 0.d0
   allocate(overlaps(totalsizes(itrain),totalsizes(itrain)))
   overlaps(:,:) = 0.d0
   allocate(kernels(kernsizes(itrain)))
   kernels(:) = 0.d0
   allocate(sparseindexes(2*llmax+1,natoms(itrain),nnmax,llmax+1))
   sparseindexes(:,:,:,:) = 0
   allocate(kernsparseindexes(2*llmax+1,natoms(itrain),2*llmax+1,llmax+1,M))
   kernsparseindexes(:,:,:,:,:) = 0
   ! read projections, overlaps and kernels for each training molecule
   conf = trrange(itrain)
   write(conf_str,*) conf
   write(ref_str,*) M 
   if1 = itrain + 10
   if2 = itrain + 2000 
   if3 = itrain + 4000
   open(unit=if1,file=trim(adjustl(dirprojs))//'projections_conf'//trim(adjustl(conf_str))//'.dat',action='read',status='old')
   open(unit=if2,file=trim(adjustl(dirover))//'overlap_conf'//trim(adjustl(conf_str))//'.dat',action='read',status='old')
   it1 = 1
   do iat=1,natoms(itrain)
      a1 = atomspe(itrain,iat)+1
      al1 = almax(a1)
      do l1=1,al1
         msize1 = 2*(l1-1)+1
         anc1 = ancut(a1,l1)
         do n1=1,anc1
            do im1=1,msize1
               read(if1,*) proj 
               projections(it1) = proj
               it2 = 1 
               do jat=1,natoms(itrain)
                  a2 = atomspe(itrain,jat)+1
                  al2 = almax(a2)
                  do l2=1,al2
                     msize2 = 2*(l2-1)+1
                     anc2 = ancut(a2,l2)
                     do n2=1,anc2
                        do im2=1,msize2
                           read(if2,*) over
                           overlaps(it2,it1) = over
                           it2 = it2 + 1
                        enddo
                     enddo
                  enddo
               enddo
               sparseindexes(im1,iat,n1,l1) = it1
               it1 = it1 + 1
            enddo
         enddo
      enddo
   enddo
   close(if2)
   close(if1)
   open(unit=if3,file=trim(adjustl(dirker))//'kernel_conf'//trim(adjustl(conf_str))//'.dat',action='read',status='old')
   ik1 = 1
   do iref1=1,M
      a1 = specarray(iref1) + 1
      al1 = almax(a1)
      do l1=1,al1
         msize1 = 2*(l1-1)+1
         do im1=1,msize1
            do iat=1,atomcount(itrain,a1)
               do imm1=1,msize1
                  read(if3,*) kern 
                  kernels(ik1) = kern
                  kernsparseindexes(imm1,iat,im1,l1,iref1) = ik1 
                  ik1 = ik1 + 1 
               enddo
            enddo
         enddo
      enddo
   enddo
   close(if3)
   ! Loop over 1st dimension
   i1 = 1
   do iref1=1,M
       a1 = specarray(iref1) + 1
       al1 = almax(a1)
       do l1=1,al1
           msize1 = 2*(l1-1)+1
           anc1 = ancut(a1,l1)
           do n1=1,anc1
               do im1=1,msize1
                   ! Collect contributions for 1st dimension 
                   do icspe1=1,atomcount(itrain,a1)
                       iat = atomicindx(icspe1,a1,itrain) + 1
                       do imm1=1,msize1
                           sp1 = sparseindexes(imm1,iat,n1,l1)
                           sk1 = kernsparseindexes(imm1,icspe1,im1,l1,iref1)
                           contrA = projections(sp1) * kernels(sk1) 
                           !$OMP ATOMIC UPDATE
                           Avec(i1) = Avec(i1) + contrA 
                           !$OMP END ATOMIC
                       enddo
                   enddo                   
                   ! Loop over 2nd dimension
                   i2 = 1
                   do iref2=1,iref1
                       a2 = specarray(iref2) + 1 
                       al2 = almax(a2)
                       do l2=1,al2
                           msize2 = 2*(l2-1)+1
                           anc2 = ancut(a2,l2)
                           do n2=1,anc2
                               do im2=1,msize2
                                   ! Collect contributions for 1st dimension
                                   contrB = 0.d0
                                   do icspe1=1,atomcount(itrain,a1)
                                       iat = atomicindx(icspe1,a1,itrain) + 1
                                       do imm1=1,msize1
                                           sp1 = sparseindexes(imm1,iat,n1,l1)
                                           sk1 = kernsparseindexes(imm1,icspe1,im1,l1,iref1)
                                           ! Collect contributions for 2nd dimension 
                                           Btemp = 0.d0
                                           do icspe2=1,atomcount(itrain,a2)
                                               jat = atomicindx(icspe2,a2,itrain) + 1
                                               do imm2=1,msize2
                                                   sp2 = sparseindexes(imm2,jat,n2,l2)
                                                   sk2 = kernsparseindexes(imm2,icspe2,im2,l2,iref2)
                                                   Btemp = Btemp + overlaps(sp2,sp1) * kernels(sk2)
                                               enddo
                                           enddo
                                           contrB = contrB + Btemp * kernels(sk1)
                                       enddo
                                   enddo
                                   !$OMP ATOMIC UPDATE
                                   Bmat(i1,i2) = Bmat(i1,i2) + contrB 
                                   !$OMP END ATOMIC
                                   if (iref2.lt.iref1) then
                                       !$OMP ATOMIC UPDATE
                                       Bmat(i2,i1) = Bmat(i2,i1) + contrB 
                                       !$OMP END ATOMIC
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
   deallocate(kernsparseindexes)
   deallocate(sparseindexes)
   deallocate(projections)
   deallocate(overlaps)
   deallocate(kernels)
enddo
!$OMP END DO
!$OMP END PARALLEL

return
END 
