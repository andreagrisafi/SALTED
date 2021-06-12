subroutine neighlist_ewald(natmax,nat,nspecies,neighmax,coords,ncell,cell,invcell,   &
                rcut2,ncentype,centers,atomidx,allspe,nneightype,neighx,coordx)

implicit none
integer:: iat,icen,cen,ispe,ncentype,icentype,centype
integer:: n,nat,nspecies,spe,ineigh,ia,ib,ic,neigh,neighmax,natmax
integer,dimension(3):: ncell
real*8:: rx,ry,rz,sx,sy,sz,rcx,rcy,rcz,x,y,z,rcut2,r2
real*8,dimension(3,3):: cell,invcell
real*8,dimension(3,nat):: coords
integer,dimension(ncentype):: centers
integer,dimension(nspecies):: allspe
integer,dimension(119):: nneightype
integer,dimension(119,natmax):: atomidx
integer,dimension(nat,nspecies):: neighx
real*8,dimension(3,neighmax,nspecies,nat):: coordx

!f2py intent(in) natmax,nat,nspecies,neighmax,coords,ncell,cell,invcell,rcut2,ncentype,centers,atomidx,allspe,nneightype 
!f2py intent(out) neighx,coordx
!f2py depend(nat) coordx,neighx,coords
!f2py depend(nspecies) coordx,neighx,allspe
!f2py depend(neighmax) coordx
!f2py depend(ncentype) centers
!f2py depend(natmax) atomidx

coordx = 0.d0
neighx = 0
iat = 1
do icentype=1,ncentype
   centype = centers(icentype) + 1
   do icen=1,nneightype(centype)
      cen = atomidx(centype,icen) + 1
      do ispe=1,nspecies
         spe = allspe(ispe) + 1
         n = 1
         do ineigh=1,nneightype(spe)
            neigh = atomidx(spe,ineigh) + 1
            rx = coords(1,neigh) - coords(1,cen)
            ry = coords(2,neigh) - coords(2,cen)
            rz = coords(3,neigh) - coords(3,cen)
            sx = invcell(1,1)*rx + invcell(1,2)*ry + invcell(1,3)*rz
            sy = invcell(2,1)*rx + invcell(2,2)*ry + invcell(2,3)*rz
            sz = invcell(3,1)*rx + invcell(3,2)*ry + invcell(3,3)*rz
            sx = sx - 1.d0*anint(sx)
            sy = sy - 1.d0*anint(sy)
            sz = sz - 1.d0*anint(sz)
            rcx = cell(1,1)*sx + cell(1,2)*sy + cell(1,3)*sz
            rcy = cell(2,1)*sx + cell(2,2)*sy + cell(2,3)*sz
            rcz = cell(3,1)*sx + cell(3,2)*sy + cell(3,3)*sz
            ! replicate cell
            do ia=1,2*ncell(1)+1
               do ib=1,2*ncell(2)+1
                  do ic=1,2*ncell(3)+1
                     x = rcx + (ia-ncell(1)-1)*cell(1,1) + (ib-ncell(2)-1)*cell(1,2) + (ic-ncell(3)-1)*cell(1,3)
                     y = rcy + (ia-ncell(1)-1)*cell(2,1) + (ib-ncell(2)-1)*cell(2,2) + (ic-ncell(3)-1)*cell(2,3)
                     z = rcz + (ia-ncell(1)-1)*cell(3,1) + (ib-ncell(2)-1)*cell(3,2) + (ic-ncell(3)-1)*cell(3,3)
                     r2 = x**2 + y**2 + z**2
                     if (r2.le.rcut2) then
                        coordx(1,n,ispe,iat) = x
                        coordx(2,n,ispe,iat) = y
                        coordx(3,n,ispe,iat) = z
                        neighx(iat,ispe) = neighx(iat,ispe) + 1
                        n = n + 1
                     endif
                  end do
               end do
            end do
         end do
      end do
      iat = iat + 1
   end do
end do

return
end

