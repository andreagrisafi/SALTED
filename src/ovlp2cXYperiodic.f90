SUBROUTINE ovlp2c(ncart1,ncart2,coord1,coord2,cell,repmax,&
                  alpha1,alpha2,cartidx1,cartidx2,overlap)

!use omp_lib
IMPLICIT NONE
! allocate I/O variables
INTEGER:: ncart1,ncart2
INTEGER, DIMENSION(3):: repmax 
INTEGER, DIMENSION(3,ncart1):: cartidx1 
INTEGER, DIMENSION(3,ncart2):: cartidx2
REAL*8:: alpha1,alpha2
REAL*8, DIMENSION(3):: coord1,coord2 
REAL*8, DIMENSION(3,3):: cell
REAL*8, DIMENSION(ncart2,ncart1):: overlap
! allocate internal variables
REAL*8, DIMENSION(3):: coord,s
INTEGER:: ix,iy,iz,ixx,iyy,izz
INTEGER:: itriplet1,itriplet2,k,k1,k2,i,ksum
INTEGER,DIMENSION(3):: triplet1,triplet2
REAL*8:: rr1,rr2,asum,s2,kab,pi,ss,argexp
REAL*8:: factx1,facty1,factz1,factx2,facty2,factz2
REAL*8:: factdiff1,factdiff2,factk1,factk2,dfact
REAL*8:: Ifactx,Ifacty,Ifactz,afact

!f2py intent(in) ncart1,ncart2,coord1,coord2,cell,repmax 
!f2py intent(in) alpha1,alpha2,cartidx1,cartidx2 
!f2py intent(out) overlap
!f2py depend(ncart1) overlap,cartidx1
!f2py depend(ncart2) overlap,cartidx2

pi = 4.d0*datan(1.d0)

rr1 = dot_product(coord1,coord1)
asum = alpha1+alpha2

overlap = 0.d0

!!$OMP PARALLEL DEFAULT(private) &
!!$OMP FIRSTPRIVATE(ncart1,ncart2,coord1,coord2,cell,repmax) &
!!$OMP FIRSTPRIVATE(pi,rr1,asum,alpha1,alpha2,cartidx1,cartidx2) &
!!$OMP SHARED(overlap)
!!$OMP DO SCHEDULE(dynamic)
do itriplet1=1,ncart1
   triplet1(:) = cartidx1(:,itriplet1)
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
   do ix=1,2*repmax(1)+1
      ixx = ix-1-repmax(1)
      do iy=1,2*repmax(2)+1
         iyy = iy-1-repmax(2)
         coord(1) = coord2(1) + ixx*cell(1,1)
         coord(2) = coord2(2) + iyy*cell(2,2)
         coord(3) = coord2(3) 
         rr2 = dot_product(coord,coord)
         s2 = alpha1*rr1 + alpha2*rr2
         s2 = s2 / asum
         s(1) = alpha1*coord1(1) + alpha2*coord(1)
         s(2) = alpha1*coord1(2) + alpha2*coord(2)
         s(3) = alpha1*coord1(3) + alpha2*coord(3)
         s(:) = s(:) / asum
         ss = dot_product(s,s)
         argexp = asum*(ss-s2)
         if (argexp.gt.-30.d0) then
            kab = dexp(argexp)
            do itriplet2=1,ncart2
               triplet2(:) = cartidx2(:,itriplet2)
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
               do k1=0,triplet1(1)
                  factdiff1 = 1.d0
                  do i = 2,triplet1(1)-k1
                     factdiff1 = factdiff1 * i
                  enddo
                  factk1 = 1.d0
                  do i = 2, k1
                     factk1 = factk1 * i
                  enddo
                  do k2=0,triplet2(1)
                     factdiff2 = 1.d0
                     do i = 2,triplet2(1)-k2
                        factdiff2 = factdiff2 * i
                     enddo
                     factk2 = 1.d0
                     do i = 2, k2
                        factk2 = factk2 * i
                     enddo
                     ksum = k1+k2
                     if (mod(ksum,2).eq.0) then
                        dfact = 1.d0
                        if ((ksum-1).gt.0) then
                           do i = 2,ksum/2
                              dfact = dfact * (2*i-1)
                           enddo
                        endif
                        Ifactx = Ifactx + &
                        (s(1)-coord1(1))**(triplet1(1)-k1) * (s(1)-coord(1))**(triplet2(1)-k2) * &
                        factx1/(factdiff1*factk1) * factx2/(factdiff2*factk2) * &
                        dsqrt(pi/asum) * dfact / (2.d0*asum)**(ksum/2)
                     endif
                  enddo
               enddo
               ! Compute integral over y 
               Ifacty = 0.d0
               do k1=0,triplet1(2)
                  factdiff1 = 1.d0
                  do i = 2,triplet1(2)-k1
                     factdiff1 = factdiff1 * i
                  enddo
                  factk1 = 1.d0
                  do i = 2, k1
                     factk1 = factk1 * i
                  enddo
                  do k2=0,triplet2(2)
                     factdiff2 = 1.d0
                     do i = 2,triplet2(2)-k2
                        factdiff2 = factdiff2 * i
                     enddo
                     factk2 = 1.d0
                     do i = 2, k2
                        factk2 = factk2 * i
                     enddo
                     ksum = k1+k2
                     if (mod(ksum,2).eq.0) then
                        dfact = 1.d0
                        if ((ksum-1).gt.0) then
                           do i = 2,ksum/2
                              dfact = dfact * (2*i-1)
                           enddo
                        endif
                        Ifacty = Ifacty + &
                        (s(2)-coord1(2))**(triplet1(2)-k1) * (s(2)-coord(2))**(triplet2(2)-k2) * &
                        facty1/(factdiff1*factk1) * facty2/(factdiff2*factk2) * &
                        dsqrt(pi/asum) * dfact / (2.d0*asum)**(ksum/2)
                     endif
                  enddo
               enddo
               ! Compute integral over z 
               Ifactz = 0.d0
               do k1=0,triplet1(3)
                  factdiff1 = 1.d0
                  do i = 2,triplet1(3)-k1
                     factdiff1 = factdiff1 * i
                  enddo
                  factk1 = 1.d0
                  do i = 2, k1
                     factk1 = factk1 * i
                  enddo
                  do k2=0,triplet2(3)
                     factdiff2 = 1.d0
                     do i = 2,triplet2(3)-k2
                        factdiff2 = factdiff2 * i
                     enddo
                     factk2 = 1.d0
                     do i = 2, k2
                        factk2 = factk2 * i
                     enddo
                     ksum = k1+k2
                     if (mod(ksum,2).eq.0) then
                        dfact = 1.d0
                        if ((ksum-1).gt.0) then
                           do i = 2,ksum/2
                              dfact = dfact * (2*i-1)
                           enddo
                        endif
                        Ifactz = Ifactz + &
                        (s(3)-coord1(3))**(triplet1(3)-k1) * (s(3)-coord(3))**(triplet2(3)-k2) * &
                        factz1/(factdiff1*factk1) * factz2/(factdiff2*factk2) * &
                        dsqrt(pi/asum) * dfact / (2.d0*asum)**(ksum/2)
                     endif
                  enddo
               enddo
               overlap(itriplet2,itriplet1) = overlap(itriplet2,itriplet1) +  kab * Ifactx * Ifacty * Ifactz
            enddo
         endif
      enddo
   enddo
enddo
!!$OMP END DO
!!$OMP END PARALLEL

return
END 
