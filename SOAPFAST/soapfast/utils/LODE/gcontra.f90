subroutine gcontra(nat,nspecies,nmax,lmax,nG,orthoradint,harmonic,phase,omega) 

! This routine performes the G-contraction needed to get the Poisson-Rayleigh projections <anlm|V_i> 

implicit none
integer:: iG,iat,ispe,n,l,im,lm
integer:: nG,lmax,nmax,nat,nspecies
real*8,dimension(lmax+1,nmax,nG):: orthoradint 
complex*16,dimension(2,nspecies,nat,nG):: phase
complex*16,dimension((lmax+1)*(lmax+1),nG):: harmonic 
complex*16,dimension(nat,nspecies,nmax,lmax+1,2*lmax+1):: omega

!f2py intent(in) nat,nspecies,nmax,lmax,nG,orthoradint,harmonic,phase
!f2py intent(out) omega 
!f2py depend(nat) phase,omega 
!f2py depend(nspecies) phase,omega 
!f2py depend(nmax) orthoradint,omega 
!f2py depend(lmax) orthoradint,harmonic,omega
!f2py depend(nG) orthoradint,harmonic,phase

omega = dcmplx(0.d0,0.d0)
!$OMP PARALLEL DEFAULT(private) &
!$OMP SHARED(nat,nspecies,nmax,lmax,nG,orthoradint,harmonic,phase,omega)
!$OMP DO SCHEDULE(dynamic)
do iat=1,nat
   do iG=2,nG
      do ispe=1,nspecies
         do n=1,nmax
            lm = 1
            do l=1,lmax+1
               if (mod(l-1,2).eq.0) then
                   do im=1,2*(l-1)+1
                      omega(iat,ispe,n,l,im) = omega(iat,ispe,n,l,im) + harmonic(lm,iG) * orthoradint(l,n,iG) * phase(1,ispe,iat,iG)
                      lm = lm + 1
                   end do
                else
                   do im=1,2*(l-1)+1
                      omega(iat,ispe,n,l,im) = omega(iat,ispe,n,l,im) + harmonic(lm,iG) * orthoradint(l,n,iG) * phase(2,ispe,iat,iG)
                      lm = lm + 1
                   end do
                end if 
            end do
         end do
      end do
   end do
end do
!$OMP END DO
!$OMP END PARALLEL

return
end

