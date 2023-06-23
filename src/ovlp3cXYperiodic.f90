SUBROUTINE ovlp3c(ncart,naocart,natoms,species,coords,cell,&
                  nspecies,llaomax,laomax,naomax,npgf,rr,alpha,&
                  npgfmax,aoalphas,iat,l,llmax,cartidx,nnaomax,contra,&
                  rc,aorcuts,overlap)
!use omp_lib
IMPLICIT NONE
! allocate I/O variables
INTEGER:: ncart,naocart,natoms,nspecies,llaomax,npgfmax,iat,l,llmax,nnaomax
INTEGER, DIMENSION(natoms):: species 
INTEGER, DIMENSION(nspecies):: laomax,npgf 
INTEGER, DIMENSION(nspecies,llaomax+1):: naomax 
INTEGER, DIMENSION(3,(llmax+1)*(llmax+2)/2,llmax+1):: cartidx 
REAL*8:: rr,alpha,rc
REAL*8, DIMENSION(nspecies,npgfmax):: aoalphas
REAL*8, DIMENSION(npgfmax,llaomax+1,nspecies):: aorcuts 
REAL*8, DIMENSION(npgfmax,nnaomax,llaomax+1,nspecies):: contra 
REAL*8, DIMENSION(3,natoms):: coords 
REAL*8, DIMENSION(3,3):: cell
REAL*8, DIMENSION(naocart,naocart,ncart):: overlap
! allocate internal variables
INTEGER:: ix1,iy1,iz1,ixx1,iyy1,izz1,iaocart1,iat1,spe1,l1,n1,ncart1,ipgf1 
INTEGER:: ix2,iy2,iz2,ixx2,iyy2,izz2,iaocart2,iat2,spe2,l2,n2,ncart2,ipgf2 
INTEGER:: itriplet,itriplet1,itriplet2,k,k1,k2,i,ksum,iao1,iao2
INTEGER,DIMENSION(3):: triplet,triplet1,triplet2,repmax1,repmax2
REAL*8,DIMENSION(3):: coord1,coord2,s,sp
REAL*8:: rr1,rr2,asum,s2,ss,kabc,pi,work
REAL*8:: factx,facty,factz,factx1,facty1,factz1,factx2,facty2,factz2
REAL*8:: factdiff,factdiff1,factdiff2,factk,factk1,factk2,dfact
REAL*8:: Ifactx,Ifacty,Ifactz,argexp,s2p,dmax,contra1,contra2
REAL*8, DIMENSION(:,:), ALLOCATABLE:: ovlp_temp

!f2py intent(in) ncart, naocart, natoms, species, coords, cell
!f2py intent(in) nspecies,llaomax,laomax,naomax,npgf,rr,alpha
!f2py intent(in) npgfmax,aoalphas,iat,l,llmax,cartidx,nnaomax,contra,rc,aorcuts
!f2py intent(out) overlap
!f2py depend(ncart) overlap
!f2py depend(naocart) overlap
!f2py depend(natoms) species,coords
!f2py depend(nspecies) laomax,naomax,npgf,aoalphas,contra,aorcuts 
!f2py depend(llaomax) naomax,contra,aorcuts 
!f2py depend(nnaomax) contra
!f2py depend(npgfmax) aoalphas,contra,aorcuts
!f2py depend(llmax) cartidx 

pi = 4.d0*datan(1.d0)

! init overlap of size ((1+l)*(2+l)/2,naocart,naocart)
overlap = 0.d0

!$OMP PARALLEL DEFAULT(private) &
!$OMP FIRSTPRIVATE(ncart,naocart,natoms,species,coords,cell,pi) &
!$OMP FIRSTPRIVATE(nspecies,llaomax,laomax,naomax,npgf,rr,alpha) &
!$OMP FIRSTPRIVATE(npgfmax,aoalphas,iat,l,llmax,cartidx,nnaomax,contra,rc,aorcuts) &
!$OMP SHARED(overlap)
!$OMP DO SCHEDULE(dynamic)
do itriplet=1,ncart
   triplet(:) = cartidx(:,itriplet,l+1)
   factx = 1.d0
   do i = 2, triplet(1)
     factx = factx * i
   enddo
   facty = 1.d0
   do i = 2, triplet(2)
     facty = facty * i
   enddo
   factz = 1.d0
   do i = 2, triplet(3)
     factz = factz * i
   enddo
   allocate(ovlp_temp(naocart,naocart))
   ovlp_temp(:,:) = 0.d0
   ! for each atom of the first AO
   iaocart1 = 0
   do iat1=1,natoms
      spe1 = species(iat1)
      do l1=0,laomax(spe1)
         ncart1 = (2+l1)*(1+l1)/2
         do n1=1,naomax(spe1,l1+1)
            do ipgf1=1,npgf(spe1)
               contra1 = contra(ipgf1,n1,l1+1,spe1) 
               dmax = rc+aorcuts(ipgf1,l1+1,spe1)
               repmax1(1) = ceiling(dmax/cell(1,1))
               repmax1(2) = ceiling(dmax/cell(2,2))
               repmax1(3) = ceiling(dmax/cell(3,3))
               ! for each periodic image of the first AO
               do ix1=1,2*repmax1(1)+1
                  ixx1 = ix1-1-repmax1(1)
                  do iy1=1,2*repmax1(2)+1
                     iyy1 = iy1-1-repmax1(2)
                     coord1(:) = 0.d0
                     coord1(1) = coords(1,iat1) + ixx1*cell(1,1)
                     coord1(2) = coords(2,iat1) + iyy1*cell(2,2)
                     coord1(3) = coords(3,iat1) 
                     rr1 = dot_product(coord1,coord1)
                     s2p = alpha*rr+aoalphas(spe1,ipgf1)*rr1 
                     sp(:) = 0.d0
                     sp(1) = alpha*coords(1,iat+1)+aoalphas(spe1,ipgf1)*coord1(1)
                     sp(2) = alpha*coords(2,iat+1)+aoalphas(spe1,ipgf1)*coord1(2)
                     sp(3) = alpha*coords(3,iat+1)+aoalphas(spe1,ipgf1)*coord1(3)
                     ! for each atom of the second AO
                     iaocart2 = 0
                     do iat2=1,natoms
                        spe2 = species(iat2)
                        do l2=0,laomax(spe2)
                           ncart2 = (2+l2)*(1+l2)/2
                           do n2=1,naomax(spe2,l2+1)
                              do ipgf2=1,npgf(spe2)
                                 contra2 = contra(ipgf2,n2,l2+1,spe2) 
                                 dmax = rc+aorcuts(ipgf2,l2+1,spe2)
                                 repmax2(1) = ceiling(dmax/cell(1,1))
                                 repmax2(2) = ceiling(dmax/cell(2,2))
                                 repmax2(3) = ceiling(dmax/cell(3,3))
                                 asum = alpha+aoalphas(spe1,ipgf1)+aoalphas(spe2,ipgf2)
                                 ! for each periodic image of the second AO
                                 do ix2=1,2*repmax2(1)+1
                                    ixx2 = ix2-1-repmax2(1)
                                    do iy2=1,2*repmax2(2)+1
                                       iyy2 = iy2-1-repmax2(2)
                                       coord2(:) = 0.d0
                                       coord2(1) = coords(1,iat2) + ixx2*cell(1,1)
                                       coord2(2) = coords(2,iat2) + iyy2*cell(2,2)
                                       coord2(3) = coords(3,iat2) 
                                       rr2 = dot_product(coord2,coord2)
                                       s2 = s2p+aoalphas(spe2,ipgf2)*rr2
                                       s2 = s2 / asum 
                                       s(:) = 0.d0
                                       s(1) = sp(1)+aoalphas(spe2,ipgf2)*coord2(1)
                                       s(2) = sp(2)+aoalphas(spe2,ipgf2)*coord2(2)
                                       s(3) = sp(3)+aoalphas(spe2,ipgf2)*coord2(3)
                                       s(:) = s(:) / asum
                                       ss = dot_product(s,s)
                                       argexp = asum*(ss-s2)
                                       ! proceed only if there is a non-negligible 3-center overlap
                                       if (argexp.gt.-30.d0) then
                                          kabc = dexp(argexp)
                                          ! for each Cartesian triplet of the first AO
                                          do itriplet1=1,ncart1
                                             iao1 = iaocart1+itriplet1
                                             triplet1(:) = cartidx(:,itriplet1,l1+1) 
                                             factx1 = 1.d0
                                             do i = 2, triplet1(1)
                                                factx1 = factx1 * i
                                             enddo 
                                             facty1 = 1.d0
                                             do i = 2, triplet1(2)
                                               facty1 = facty1 * i
                                             enddo 
                                             factz1 = 1.d0
                                             do i = 2, triplet1(3)
                                               factz1 = factz1 * i
                                             enddo 
                                             ! for each Cartesian triplet of the second AO
                                             do itriplet2=1,ncart2
                                                iao2 = iaocart2+itriplet2
                                                triplet2(:) = cartidx(:,itriplet2,l2+1)
                                                factx2 = 1.d0
                                                do i = 2, triplet2(1)
                                                  factx2 = factx2 * i
                                                enddo 
                                                facty2 = 1.d0
                                                do i = 2, triplet2(2)
                                                  facty2 = facty2 * i
                                                enddo 
                                                factz2 = 1.d0
                                                do i = 2, triplet2(3)
                                                  factz2 = factz2 * i
                                                enddo
                                                ! Compute integral over x 
                                                Ifactx = 0.d0
                                                do k=0,triplet(1)
                                                   factdiff = 1.d0
                                                   do i = 2, triplet(1)-k
                                                      factdiff = factdiff * i
                                                   enddo
                                                   factk = 1.d0
                                                   do i = 2, k
                                                      factk = factk * i
                                                   enddo
                                                   do k1=0,triplet1(1)
                                                      factdiff1 = 1.d0
                                                      do i = 2, triplet1(1)-k1
                                                         factdiff1 = factdiff1 * i
                                                      enddo
                                                      factk1 = 1.d0
                                                      do i = 2, k1
                                                         factk1 = factk1 * i
                                                      enddo
                                                      do k2=0,triplet2(1)
                                                         factdiff2 = 1.d0
                                                         do i = 2, triplet2(1)-k2
                                                            factdiff2 = factdiff2 * i
                                                         enddo
                                                         factk2 = 1.d0
                                                         do i = 2, k2
                                                            factk2 = factk2 * i
                                                         enddo
                                                         ksum = k+k1+k2
                                                         if(mod(ksum,2).eq.0) then
                                                            dfact = 1.d0
                                                            if((ksum-1).gt.0) then
                                                                do i = 2, ksum/2
                                                                   dfact = dfact * (2*i-1)
                                                                enddo
                                                            endif
                                                            Ifactx = Ifactx + &  
       (s(1)-coords(1,iat+1))**(triplet(1)-k) * (s(1)-coord1(1))**(triplet1(1)-k1) * (s(1)-coord2(1))**(triplet2(1)-k2) * &
      factx/(factdiff*factk) * factx1/(factdiff1*factk1) * factx2/(factdiff2*factk2)  * dsqrt(pi/asum) * &
      dfact / (2.d0*asum)**(ksum/2) 
                                                         endif
                                                      enddo
                                                   enddo
                                                enddo
                                                ! Compute integral over y 
                                                Ifacty = 0.d0
                                                do k=0,triplet(2)
                                                   factdiff = 1.d0
                                                   do i = 2, triplet(2)-k
                                                      factdiff = factdiff * i
                                                   enddo
                                                   factk = 1.d0
                                                   do i = 2, k
                                                      factk = factk * i
                                                   enddo
                                                   do k1=0,triplet1(2)
                                                      factdiff1 = 1.d0
                                                      do i = 2, triplet1(2)-k1
                                                         factdiff1 = factdiff1 * i
                                                      enddo
                                                      factk1 = 1.d0
                                                      do i = 2, k1
                                                         factk1 = factk1 * i
                                                      enddo
                                                      do k2=0,triplet2(2)
                                                         factdiff2 = 1.d0
                                                         do i = 2, triplet2(2)-k2
                                                            factdiff2 = factdiff2 * i
                                                         enddo
                                                         factk2 = 1.d0
                                                         do i = 2, k2
                                                            factk2 = factk2 * i
                                                         enddo
                                                         ksum = k+k1+k2
                                                         if (mod(ksum,2).eq.0) then
                                                            dfact = 1.d0
                                                            if((ksum-1).gt.0) then
                                                                do i = 2, ksum/2
                                                                   dfact = dfact * (2*i-1)
                                                                enddo
                                                            endif
                                                            Ifacty = Ifacty + &  
       (s(2)-coords(2,iat+1))**(triplet(2)-k) * (s(2)-coord1(2))**(triplet1(2)-k1) * (s(2)-coord2(2))**(triplet2(2)-k2) * &
      facty/(factdiff*factk) * facty1/(factdiff1*factk1) * facty2/(factdiff2*factk2) * dsqrt(pi/asum) * &
      dfact / (2.d0*asum)**(ksum/2)  
                                                         endif
                                                      enddo
                                                   enddo
                                                enddo
                                                ! Compute integral over z 
                                                Ifactz = 0.d0
                                                do k=0,triplet(3)
                                                   factdiff = 1.d0
                                                   do i = 2, triplet(3)-k
                                                      factdiff = factdiff * i
                                                   enddo
                                                   factk = 1.d0
                                                   do i = 2, k
                                                      factk = factk * i
                                                   enddo
                                                   do k1=0,triplet1(3)
                                                      factdiff1 = 1.d0
                                                      do i = 2, triplet1(3)-k1
                                                         factdiff1 = factdiff1 * i
                                                      enddo
                                                      factk1 = 1.d0
                                                      do i = 2, k1
                                                         factk1 = factk1 * i
                                                      enddo
                                                      do k2=0,triplet2(3)
                                                         factdiff2 = 1.d0
                                                         do i = 2, triplet2(3)-k2
                                                           factdiff2 = factdiff2 * i
                                                         enddo
                                                         factk2 = 1.d0
                                                         do i = 2, k2
                                                            factk2 = factk2 * i
                                                         enddo
                                                         ksum = k+k1+k2
                                                         if (mod(ksum,2).eq.0) then
                                                            dfact = 1.d0
                                                            if((ksum-1).gt.0) then
                                                                do i = 2, ksum/2
                                                                   dfact = dfact * (2*i-1)
                                                                enddo
                                                            endif
                                                            Ifactz = Ifactz + &  
       (s(3)-coords(3,iat+1))**(triplet(3)-k) * (s(3)-coord1(3))**(triplet1(3)-k1) * (s(3)-coord2(3))**(triplet2(3)-k2) * &
   factz/(factdiff*factk) * factz1/(factdiff1*factk1) * factz2/(factdiff2*factk2) * dsqrt(pi/asum) * & 
    dfact / (2.d0*asum)**(ksum/2) 
                                                         endif
                                                      enddo
                                                   enddo
                                                enddo 
          ovlp_temp(iao2,iao1) = ovlp_temp(iao2,iao1) + contra1 * contra2 * kabc * Ifactx * Ifacty * Ifactz 
                                             enddo ! for each triplet of the second AO
                                          enddo ! for each triplet of the first AO
                                       endif ! if non-negligible 3-center overlap 
                                    enddo ! for iy of 2nd AO
                                 enddo ! for ix of 2nd AO
                              enddo ! for each primitive GTO of the second AO
                              iaocart2 = iaocart2 + ncart2
                           enddo ! for n of 2nd AO
                        enddo ! for l of 2nd AO
                     enddo ! for atom of 2nd AO
                  enddo ! for iy of 1st AO
               enddo ! for ix of 1st AO
            enddo ! for each primitive GTO of the first AO
            iaocart1 = iaocart1 + ncart1
         enddo ! for n of 1st AO
      enddo ! for l of 1st AO
   enddo ! for atom of 1st AO
   ! fill final 3-center overlap
   do iaocart1=1,naocart
      do iaocart2=1,naocart
         overlap(iaocart2,iaocart1,itriplet) = ovlp_temp(iaocart2,iaocart1)
      enddo
   enddo
   deallocate(ovlp_temp)
enddo ! for each triplet of the auxiliary function
!$OMP END DO
!$OMP END PARALLEL

return
END 
