subroutine ggen(b1,b2,b3,nr,Gcut,G,Gx,iGx,iGmx,nG,igma)

! Generate wave-vectors in the semi-space x>0 (only orthorombic cells are treated at the moment!) 

implicit none
real*8:: Gcut2,ggx,ggy,ggz,gg2,pi
real*8, intent(in), dimension(3):: b1,b2,b3
integer, parameter:: ngx=99999999,nmax=200
integer:: i1,i2,i3,iG
integer,intent(out),dimension(3,10000000):: iGx,iGmx
real*8,intent(out),dimension(3,10000000):: Gx
real*8,intent(out),dimension(10000000):: G
integer,intent(out):: nG
integer,intent(out),dimension(3):: igma
integer,intent(in),dimension(3):: nr
real*8,intent(in):: Gcut
real*8, dimension(99999999):: wrk
integer, dimension(99999999):: index

pi=4.d0*datan(1.d0)

Gcut2=Gcut*Gcut
igma=0
nG=0

!------------------------------------------------
!       The origin: x=0, y=0, z=0

nG = nG + 1
G(nG) = 0.d0
Gx(1,nG) = 0.d0
Gx(2,nG) = 0.d0
Gx(3,nG) = 0.d0
iGx(1,nG) = 0
iGx(2,nG) = 0
iGx(3,nG) = 0
iGmx(1,nG) = 0
iGmx(2,nG) = 0
iGmx(3,nG) = 0
 
!----------------------------------------------
!       The line: x=0, y=0, z>0

do i3=1,nmax
   ggx = i3*b3(1)
   ggy = i3*b3(2)
   ggz = i3*b3(3)
   gg2 = ggx*ggx + ggy*ggy + ggz*ggz
   if (gg2.lt.Gcut2) then
      nG = nG + 1
      G(nG) = dsqrt(gg2)
      Gx(1,nG) = ggx
      Gx(2,nG) = ggy
      Gx(3,nG) = ggz
      iGx(1,nG) = 0
      iGx(2,nG) = 0
      iGx(3,nG) = i3
      iGmx(1,nG) = 0
      iGmx(2,nG) = 0
      iGmx(3,nG) = -i3
      if (i3.gt.igma(3)) igma(3)=i3
   end if
end do

!------------------------------------------------
!            The plane: x=0, y>0

do i2=1,nmax
   do i3=-nmax,nmax
      ggx = i2*b2(1) + i3*b3(1)
      ggy = i2*b2(2) + i3*b3(2)
      ggz = i2*b2(3) + i3*b3(3)
      gg2 = ggx*ggx + ggy*ggy + ggz*ggz
      if (gg2.lt.gcut2) then
         nG = nG + 1
         G(nG) = dsqrt(gg2)
         Gx(1,nG) = ggx
         Gx(2,nG) = ggy
         Gx(3,nG) = ggz
         iGx(1,nG) = 0
         iGx(2,nG) = i2
         iGx(3,nG) = i3
         iGmx(1,nG) = 0
         iGmx(2,nG) = -i2
         iGmx(3,nG) = -i3
         if (i2.gt.igma(2)) igma(2)=i2
         if (i3.gt.igma(3)) igma(3)=i3
         if (-i3.gt.igma(3)) igma(3)=-i3
      end if
   end do
end do          

!----------------------------------------------------
!             The semi-space: x>0

do i1=1,nmax
   do i2=-nmax,nmax
      do i3=-nmax,nmax
         ggx = i1*b1(1) + i2*b2(1) + i3*b3(1)
         ggy = i1*b1(2) + i2*b2(2) + i3*b3(2)
         ggz = i1*b1(3) + i2*b2(3) + i3*b3(3)
         gg2 = ggx*ggx + ggy*ggy + ggz*ggz
         if (gg2.lt.gcut2) then
            nG = nG + 1
            if (nG.gt.nGx) then
               write(6,*) 'Error in ggen: n_G > n_Gx'
               stop
            end if
            G(nG) = dsqrt(gg2)
            Gx(1,nG) = ggx
            Gx(2,nG) = ggy
            Gx(3,nG) = ggz
            iGx(1,nG) = i1
            iGx(2,nG) = i2
            iGx(3,nG) = i3
            iGmx(1,nG) = -i1
            iGmx(2,nG) = -i2
            iGmx(3,nG) = -i3
            if (i1.gt.igma(1)) igma(1) = i1
            if (i2.gt.igma(2)) igma(2) = i2
            if (-i2.gt.igma(2)) igma(2) = -i2
            if (i3.gt.igma(3)) igma(3) = i3
            if (-i3.gt.igma(3)) igma(3) = -i3
         end if
      end do
   end do
end do        

!------------------------------------------------------------------------
! Order all the arrays to increasing the magnitude G of the wave-vectors
!------------------------------------------------------------------------

call indexx(nG,G,index)

do iG=1,nG
   wrk(iG) = G(iG)
end do
do iG=1,nG
   G(iG) = wrk(index(iG))
end do

do iG=1,nG
   wrk(iG) = Gx(1,iG)
end do
do iG=1,nG
   Gx(1,iG) = wrk(index(iG))
end do

do iG=1,nG
   wrk(iG) = Gx(2,iG)
end do
do iG=1,nG
   Gx(2,iG) = wrk(index(iG))
end do

do iG=1,nG
   wrk(iG) = Gx(3,iG)
end do
do iG=1,nG
   Gx(3,iG) = wrk(index(iG))
end do

do iG=1,nG
   wrk(iG) = iGx(1,iG)
end do
do iG=1,nG
   iGx(1,iG) = wrk(index(iG))
end do

do iG=1,nG
   wrk(iG) = iGx(2,iG)
end do
do iG=1,nG
   iGx(2,iG) = wrk(index(iG))
end do

do iG=1,nG
   wrk(iG) = iGx(3,iG)
end do
do iG=1,nG
   iGx(3,iG) = wrk(index(iG))
end do

do iG=1,nG
   wrk(iG) = iGmx(1,iG)
end do
do iG=1,nG
   iGmx(1,iG) = wrk(index(iG))
end do

do iG=1,nG
   wrk(iG) = iGmx(2,iG)
end do
do iG=1,nG
   iGmx(2,iG) = wrk(index(iG))
end do

do iG=1,nG
   wrk(iG) = iGmx(3,iG)
end do
do iG=1,nG
   iGmx(3,iG) = wrk(index(iG))
end do

!---------------------------------------------------------------
!        Arrange the wave-vectors in the cubic mesh
!---------------------------------------------------------------

do iG=1,nG
   iGx(1,iG) = iGx(1,iG) + 1
   iGmx(1,iG) = iGmx(1,iG) + 1
   if (iGmx(1,iG).le.0) iGmx(1,iG) = iGmx(1,iG) + nr(1)
   iGx(2,iG) = iGx(2,iG) + 1
   if (iGx(2,iG).le.0) iGx(2,iG) = iGx(2,iG) + nr(2)
   iGmx(2,iG) = iGmx(2,iG) + 1
   if (iGmx(2,iG).le.0) iGmx(2,iG) = iGmx(2,iG) + nr(2)
   iGx(3,iG) = iGx(3,iG) + 1
   if (iGx(3,iG).le.0) iGx(3,iG) = iGx(3,iG) + nr(3)
   iGmx(3,iG) = iGmx(3,iG) + 1
   if (iGmx(3,iG).le.0) iGmx(3,iG) = iGmx(3,iG) + nr(3)
end do

return         
end

!-------------------------------------------------------------
                subroutine indexx(nG,G,indx)
!-------------------------------------------------------------
!  Indexes the array G: outputs the array index of length nG 
!  such that G(index(j)) is in ascending order for j=1,..,nG.
!-------------------------------------------------------------

implicit real*8 (a-h,o-z)
dimension G(nG),indx(nG)

do j=1,nG
   indx(j)=j
end do
l=nG/2+1
ir=nG
10 if (l.gt.1) then
       l=l-1
       indxt = indx(l)
       q = G(indxt)
    else
       indxt = indx(ir)
       q = G(indxt)
       indx(ir) = indx(1)
       ir=ir-1
       if (ir.eq.1) then
          indx(1) = indxt
          return
       end if
    end if
i = l
j = l+l
20 if (j.le.ir) then
      if (j.lt.ir) then
         if (G(indx(j)).lt.G(indx(j+1))) j=j+1
      end if
      if (q.lt.G(indx(j))) then
         indx(i) = indx(j)
         i=j
         j=j+j
      else
         j=ir+1
      end if
      go to 20
   end if
indx(i)=indxt
go to 10

return
end
