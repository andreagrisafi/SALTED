subroutine nearfield(nat,nspecies,nmax,lmax,ngrid,nneigh,neighmax,alpha,coords,sgrid,orthorad,harmonic,weights,sigewald,omega) 

! This routine performes the G-contraction needed to get the Poisson-Rayleigh projections <anlm|V_i> 

implicit none
integer:: igrid,iat,jat,ispe,n,l,im,lm
integer:: ngrid,lmax,nmax,nat,nspecies,neighmax
real*8:: alpha,potential,r,sigewald,alphaewald
integer,dimension(nat,nspecies)::nneigh
real*8,dimension(3,neighmax,nspecies,nat):: coords
real*8,dimension(nmax,ngrid):: orthorad
real*8,dimension(ngrid):: weights 
real*8,dimension(ngrid,3):: sgrid 
complex*16,dimension((lmax+1)*(lmax+1),ngrid):: harmonic 
complex*16,dimension(2*lmax+1,lmax+1,nmax,nspecies,nat):: omega
!complex*16,dimension(nat,nspecies,nmax,lmax+1,2*lmax+1):: omega

!f2py intent(in) nat,nspecies,nmax,lmax,ngrid,nneigh,neighmax,alpha,coords,sgrid,orthorad,harmonic,weights,sigewald 
!f2py intent(out) omega 
!f2py depend(neighmax) coords 
!f2py depend(nat) nneigh,omega,coords 
!f2py depend(nspecies) nneigh,coords,omega 
!f2py depend(nmax) orthorad,omega  
!f2py depend(lmax) harmonic,omega
!f2py depend(ngrid) weights,harmonic,orthorad,sgrid 

alphaewald = 1.d0/(2.d0*sigewald**2)

omega = dcmplx(0.d0,0.d0)
!$OMP PARALLEL DEFAULT(private) &
!$OMP SHARED(nat,nspecies,nmax,lmax,ngrid,nneigh,alpha,alphaewald,coords,sgrid,orthorad,harmonic,weights,omega)
!$OMP DO SCHEDULE(dynamic)
do iat=1,nat
   do ispe=1,nspecies
      do igrid=1,ngrid
         potential = 0.d0
         do jat=1,nneigh(iat,ispe)
            r = dsqrt((sgrid(igrid,1)-coords(1,jat,ispe,iat))**2 &
                     +(sgrid(igrid,2)-coords(2,jat,ispe,iat))**2 &
                     +(sgrid(igrid,3)-coords(3,jat,ispe,iat))**2)
            potential = potential + ( erf(dsqrt(alpha)*r) - erf(dsqrt(alphaewald)*r) ) / r
         enddo
         do n=1,nmax
            lm = 1
            do l=1,lmax+1
               do im=1,2*(l-1)+1
                  !omega(iat,ispe,n,l,im) = omega(iat,ispe,n,l,im) + harmonic(lm,igrid) &
                  omega(im,l,n,ispe,iat) = omega(im,l,n,ispe,iat) + harmonic(lm,igrid) &
                                                                  * orthorad(n,igrid)  &
                                                                  * weights(igrid) & 
                                                                  * potential 
                  lm = lm + 1
               end do
            end do
         end do
      end do
   end do
end do
!$OMP END DO
!$OMP END PARALLEL

return
end

