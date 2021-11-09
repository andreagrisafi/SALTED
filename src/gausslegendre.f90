!MJW - 19/11/18
module gauss_legendre
implicit none
save

real(8), parameter :: pi = 3.14159265359d0

contains

!----------------------------------------------------------------------------------------!

!Numerical Recipes
!Gauss Legendre grid points and weights
subroutine gaulegf(x1, x2, x, w, n)
  integer, intent(in) :: n
  real(8), intent(in) :: x1, x2
  real(8), dimension(n), intent(out) :: x, w
  integer :: i, j, m
  real(8) :: p1, p2, p3, pp, xl, xm, z, z1
  real(8), parameter :: eps=3.d-14
      
  m = (n+1)/2
  xm = 0.5d0*(x2+x1)
  xl = 0.5d0*(x2-x1)
  do i=1,m
    z = cos(3.141592654d0*(i-0.25d0)/(n+0.5d0))
    z1 = 0.0
    do while(abs(z-z1) .gt. eps)
      p1 = 1.0d0
      p2 = 0.0d0
      do j=1,n
        p3 = p2
        p2 = p1
        p1 = ((2.0d0*j-1.0d0)*z*p2-(j-1.0d0)*p3)/j
      end do
      pp = n*(z*p1-p2)/(z*z-1.0d0)
      z1 = z
      z = z1 - p1/pp
    end do
    x(i) = xm - xl*z
    x(n+1-i) = xm + xl*z
    w(i) = (2.0d0*xl)/((1.0d0-z*z)*pp*pp)
    w(n+1-i) = w(i)
  end do

end subroutine gaulegf

!----------------------------------------------------------------------------------------!

!Numerical Recipes
!Associated Legendre polynomial
real(8) function plgndr(l, m, x)
  integer, intent(in) :: l, m
  real(8), intent(in) :: x
  integer :: i, ll
  real(8) :: fact, oldfact, pll, pmm, pmmp1, omx2
  real(8), parameter :: pi = 3.14159265359d0
  
  pll = 0.0d0
  !if(m.lt.0.or.m.gt.l.or.abs(x).gt.1.) pause 'bad arguments in plgndr'
  
  pmm=1.0d0
  if (m .gt. 0) then
    omx2 = (1.d0-x)*(1.d0+x)
    fact = 1.d0
    do i = 1, m
      pmm = pmm*omx2*fact/(fact+1.d0)
      fact = fact + 2.d0
    end do
  end if
  
  pmm = sqrt((2*m + 1)*pmm/(4.d0*pi))
  if (mod(m, 2) .eq. 1) pmm = -pmm
  if (l .eq. m) then
    plgndr = pmm
  else
    pmmp1 = x*sqrt(2.d0*m + 3.d0)*pmm
    if (l .eq. m+1) then
      plgndr = pmmp1
    else
      oldfact=sqrt(2.d0*m + 3.d0)
    do ll = m+2, l
      fact = sqrt((4.d0*ll**2 - 1.d0)/(ll**2-m**2))
      pll = (x*pmmp1-pmm/oldfact)*fact
      oldfact = fact
      pmm = pmmp1
      pmmp1 = pll
    end do
      plgndr = pll
    end if
  end if

end function plgndr

!----------------------------------------------------------------------------------------!

!Hyperbolic cotangent
real(8) function coth(x)
  real(8), intent(in) :: x
  
  coth = (1.0d0 + exp(-2.0d0*x))/(1.0d0 - exp(-2.0d0*x))

end function coth

!----------------------------------------------------------------------------------------!

!Hyperbolic cotangent in quad precision
real function cothquad(x)
  real, intent(in) :: x
  
  cothquad = (1.0d0 + exp(-2.0d0*x))/(1.0d0 - exp(-2.0d0*x))

end function cothquad

!----------------------------------------------------------------------------------------!

!Partial expansion
subroutine fsub(r, sigma, rij, lmax, f)
  real(8), intent(in) :: r, sigma, rij
  real(8), dimension(0:lmax), intent(out) :: f
  integer, intent(in) :: lmax
  real(8) :: alpha, frev
  real(8), parameter :: eps = 1.0d-3
  integer :: i

  alpha = rij/sigma**2

  if (abs(r - rij) >= 5.0d0*sigma) then
      f(0:lmax) = 0.0d0
      return
  end if

  if (r <= eps) then
      f(0:lmax) = 0.0d0
      return
  end if
   
  if (rij <= eps) then
      f(0) = exp(-r**2/(2.0d0*sigma**2))*r
      f(1:lmax) = 0.0d0
      return
  end if
  
  !l=0
  f(0) = exp(-0.5d0*(r - rij)**2/sigma**2)
  f(0) = f(0) - exp(-0.5d0*(r + rij)**2/sigma**2)
  f(0) = 0.5d0*f(0)/alpha

  !l=1
  if (alpha*r >= eps) then !exact
      f(1) = f(0)*(coth(alpha*r) - 1.0d0/(r*alpha))
  else !first three terms in Taylor series
      f(1) = f(0)*(alpha*r/3.0d0 - (alpha*r)**2/45.0d0 + (alpha*r)**5/945.0d0)
  end if

  !in = in-1 - (2(n-1) + 1)in-1/x
  do i = 2, lmax
      f(i) = f(i-2) - (2.0d0*dble(i - 1) + 1.0d0)*f(i-1)/(alpha*r)
      frev = f(i) + (2.0d0*dble(i - 1) + 1.0d0)*f(i-1)/(alpha*r)
      frev = f(i-1) + (2.0d0*dble(i - 2) + 1.0d0)*frev/(alpha*r)
      if (i >= 3 .and. abs(frev - f(i-3)) >= 1.0d-100) then
!        print *, 'Iteration in l terminated early because of numerical instability'
        if (f(i)/f(0) > 1.0d-8) cycle
        f(i:lmax) = 0.0d0
        exit
      end if
  end do

end subroutine fsub

!----------------------------------------------------------------------------------------!

!Partial expansion in quad precision
subroutine fsubquad(r2, sigma2, rij2, lmax, f2)
  real(kind=8), intent(in) :: r2, sigma2, rij2
  real(kind=8), dimension(0:lmax), intent(out) :: f2
  integer, intent(in) :: lmax
  real :: r, sigma, rij
  real, dimension(0:lmax) :: f
  real :: alpha, frev
  real, parameter :: eps = 1.d-5
  integer :: i

  r = real(r2)
  sigma = real(sigma2)
  rij = real(rij2)
  alpha = rij/sigma**2

  if (abs(r - rij) >= 5.d0*sigma) then
      f(0:lmax) = 0.0d0
      f2 = dble(f)
      return
  end if

  if (r <= eps) then
      f(0:lmax) = 0.0d0
      f2 = dble(f)
      return
  end if

  if (rij <= eps) then
      f(0) = exp(-r**2/(2.0d0*sigma**2))*r
      f(1:lmax) = 0.0d0
      f2 = dble(f)
      return
  end if

  !l=0
  f(0) = exp(-0.5d0*(r - rij)**2/sigma**2)
  f(0) = f(0) - exp(-0.5d0*(r + rij)**2/sigma**2)
  f(0) = 0.5d0*f(0)/alpha

  !l=1
  if (alpha*r >= eps) then !exact
      f(1) = f(0)*(cothquad(alpha*r) - 1.0d0/(r*alpha))
  else !first three terms in Taylor series
      f(1) = f(0)*(alpha*r/3.0d0 - (alpha*r)**2/45.0d0 + (alpha*r)**5/945.0d0)
  end if


  !in = in-1 - (2(n-1) + 1)in-1/x
  do i = 2, lmax
      f(i) = f(i-2) - (2.0d0*real(i - 1) + 1.0d0)*f(i-1)/(alpha*r)
      frev = f(i) + (2.0d0*real(i - 1) + 1.0d0)*f(i-1)/(alpha*r)
      frev = f(i-1) + (2.0d0*real(i - 2) + 1.0d0)*frev/(alpha*r)
      if (i >= 3 .and. abs(frev - f(i-3)) >= 1.0d-100) then
!        print *, 'Warning: iteration in l terminated early because of numerical instability'
        if (f(i)/f(0) > 1.0d-8) cycle
        f(i:lmax) = 0.0d0
        exit
      end if
  end do

  f2 = dble(f)

end subroutine fsubquad

!----------------------------------------------------------------------------------------!

!Full expansion
subroutine f2sub(f, nmax, lmax, cost, f2)
  real(8), dimension(0:nmax, 0:lmax), intent(in) :: f
  real(8), intent(in) :: cost
  integer, intent(in) :: nmax, lmax
  real(8), dimension(0:nmax, 0:lmax, 0:lmax), intent(out) :: f2
  integer :: i, j

  do i = 0, lmax
    f2(:, i, 0) = f(:, i)*plgndr(i, 0, cost)
    do j = 1, lmax
      f2(:, i, j) = f(:, i)*plgndr(i, j, cost)
    end do
  end do

end subroutine f2sub

!----------------------------------------------------------------------------------------!

subroutine coulomb(nmax, lmax, rcut, c)
!Return coulomb matrix |r - r'|^-1
  integer, intent(in) :: nmax, lmax
  real(8), intent(in) :: rcut
  real(8), dimension(0:nmax, 0:nmax, 0:lmax, 0:lmax), intent(out) :: c
  real(8), dimension(0:lmax, 0:lmax, 0:2*lmax) :: b
  real(8), dimension(0:2*lmax) :: x, w, f1, f2, f3
  integer :: i, j, k, l
  real(8), dimension(0:nmax) :: xn, wn

  call gaulegf(-1.0d0, 1.0d0, x, w, 2*lmax+1)

  do i = 0, lmax
    do l = 0, 2*lmax
      f1(l) = plgndr(i, 0, x(l))
    end do
    do j = 0, lmax
      do l = 0, 2*lmax
        f2(l) = plgndr(j, 0, x(l))
      end do
      do k = 0, 2*lmax
        do l = 0, 2*lmax
          f3(l) = plgndr(k, 0, x(l))
        end do
        b(i, j, k) = sum(f1*f2*f3*w)
        !what is the correct power of 2l + 1?
        b(i, j, k) = b(i, j, k)/sqrt((2.0d0*dble(k) + 1.0d0))
      end do
    end do
  end do

  c = 0.0d0
  call gaulegf(0.0d0, rcut, xn, wn, nmax+1)
  do i = 0, nmax
    do j = i+1, nmax
      do l = 0, 2*lmax
        c(i, j, :, :) = c(i, j, :, :) + b(:, :, l)*xn(i)**l/xn(j)**(l+1)
        c(j, i, :, :) = c(i, j, :, :)
      end do
    end do
  end do

end subroutine coulomb


end module gauss_legendre
