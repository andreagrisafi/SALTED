subroutine phasecomb(nat,nspecies,nneigh,nG,coords,Gvec,phase) 

! This routine computes the Fourier components of the 3D potential by combining the isotropic single-site contributions

implicit none
integer:: iG,i,j,ispe
real*8:: pi
real*8:: coordx,coordy,coordz,arg,Gx,Gy,Gz,cosine,sine,pidiv,pires,sig
integer:: nat,nG,nspecies
integer,dimension(nat,nspecies):: nneigh
real*8,dimension(3,nG):: Gvec
real*8,dimension(nat,nspecies,nat,3):: coords 
complex*16,dimension(2,nspecies,nat,nG):: phase

!f2py intent(in) nat,nspecies,nneigh,nG,coords,Gvec
!f2py intent(out) phase 
!f2py depend(nat) coords,phase,nneigh
!f2py depend(nspecies) coords,phase,nneigh
!f2py depend(nG) Gvec,phase

pi=4.d0*atan(1.d0)

phase = dcmplx(0.d0,0.d0)
!$OMP PARALLEL DEFAULT(private) &
!$OMP SHARED(phase,Gvec,coords,nneigh,nG,nspecies,nat)
!$OMP DO SCHEDULE(dynamic)
do iG=2,nG
   Gx = Gvec(1,iG)
   Gy = Gvec(2,iG)
   Gz = Gvec(3,iG)
   do i=1,nat
      do ispe=1,nspecies
         do j=1,nneigh(i,ispe)
            coordx = coords(i,ispe,j,1)
            coordy = coords(i,ispe,j,2)
            coordz = coords(i,ispe,j,3)
            arg = Gx*coordx + Gy*coordy + Gz*coordz
            cosine = dcos(arg)
            sine = dsin(arg)
            phase(1,ispe,i,iG) = phase(1,ispe,i,iG) + dcmplx(cosine,0.d0)
            phase(2,ispe,i,iG) = phase(2,ispe,i,iG) + dcmplx(0.d0,-sine)
         end do
      end do
   end do
end do
!$OMP END DO
!$OMP END PARALLEL

return
end

