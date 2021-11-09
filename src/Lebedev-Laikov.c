/* f2c'ed code originally from:
   http://ccl.net/cca/software/SOURCES/FORTRAN/Lebedev-Laikov-Grids/Lebedev-Laikov.F */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

static int gen_oh_(int code, int *num, double *x, double *y, double *z,
                   double *w, double *a, double *b, double *v)
{
    double c;

    /* hvd */
    /* hvd   This subroutine is part of a set of subroutines that generate */
    /* hvd   Lebedev grids [1-6] for integration on a sphere. The original */
    /* hvd   C-code [1] was kindly provided by Dr. Dmitri N. Laikov and */
    /* hvd   translated into fortran by Dr. Christoph van Wuellen. */
    /* hvd   This subroutine was translated from C to fortran77 by hand. */
    /* hvd */
    /* hvd   Users of this code are asked to include reference [1] in their */
    /* hvd   publications, and in the user- and programmers-manuals */
    /* hvd   describing their codes. */
    /* hvd */
    /* hvd   This code was distributed through CCL (http://www.ccl.net/). */
    /* hvd */
    /* hvd   [1] V.I. Lebedev, and D.N. Laikov */
    /* hvd       "A quadrature formula for the sphere of the 131st */
    /* hvd        algebraic order of accuracy" */
    /* hvd       Doklady Mathematics, Vol. 59, No. 3, 1999, pp. 477-481. */
    /* hvd */
    /* hvd   [2] V.I. Lebedev */
    /* hvd       "A quadrature formula for the sphere of 59th algebraic */
    /* hvd        order of accuracy" */
    /* hvd       Russian Acad. Sci. Dokl. Math., Vol. 50, 1995, pp. 283-286. */
    /* hvd */
    /* hvd   [3] V.I. Lebedev, and A.L. Skorokhodov */
    /* hvd       "Quadrature formulas of orders 41, 47, and 53 for the sphere" */
    /* hvd       Russian Acad. Sci. Dokl. Math., Vol. 45, 1992, pp. 587-592. */
    /* hvd */
    /* hvd   [4] V.I. Lebedev */
    /* hvd       "Spherical quadrature formulas exact to orders 25-29" */
    /* hvd       Siberian Mathematical Journal, Vol. 18, 1977, pp. 99-107. */
    /* hvd */
    /* hvd   [5] V.I. Lebedev */
    /* hvd       "Quadratures on a sphere" */
    /* hvd       Computational Mathematics and Mathematical Physics, Vol. 16, */
    /* hvd       1976, pp. 10-24. */
    /* hvd */
    /* hvd   [6] V.I. Lebedev */
    /* hvd       "Values of the nodes and weights of ninth to seventeenth */
    /* hvd        order Gauss-Markov quadrature formulae invariant under the */
    /* hvd        octahedron group with inversion" */
    /* hvd       Computational Mathematics and Mathematical Physics, Vol. 15, */
    /* hvd       1975, pp. 44-51. */
    /* hvd */
    /* vw */
    /* vw    Given a point on a sphere (specified by a and b), generate all */
    /* vw    the equivalent points under Oh symmetry, making grid points with */
    /* vw    weight v. */
    /* vw    The variable num is increased by the number of different points */
    /* vw    generated. */
    /* vw */
    /* vw    Depending on code, there are 6...48 different but equivalent */
    /* vw    points. */
    /* vw */
    /* vw    code=1:   (0,0,1) etc                                (  6 points) */
    /* vw    code=2:   (0,a,a) etc, a=1/sqrt(2)                   ( 12 points) */
    /* vw    code=3:   (a,a,a) etc, a=1/sqrt(3)                   (  8 points) */
    /* vw    code=4:   (a,a,b) etc, b=sqrt(1-2 a^2)               ( 24 points) */
    /* vw    code=5:   (a,b,0) etc, b=sqrt(1-a^2), a input        ( 24 points) */
    /* vw    code=6:   (a,b,c) etc, c=sqrt(1-a^2-b^2), a/b input  ( 48 points) */
    /* vw */
    /* Parameter adjustments */
    --w;
    --z;
    --y;
    --x;

    /* Function Body */
    switch (code) {
    case 1:
        goto L1;
    case 2:
        goto L2;
    case 3:
        goto L3;
    case 4:
        goto L4;
    case 5:
        goto L5;
    case 6:
        goto L6;
    }
    fprintf(stderr, "Gen_Oh: Invalid Code\n");
    abort();
L1:
    *a = 1.;
    x[1] = *a;
    y[1] = 0.;
    z[1] = 0.;
    w[1] = *v;
    x[2] = -(*a);
    y[2] = 0.;
    z[2] = 0.;
    w[2] = *v;
    x[3] = 0.;
    y[3] = *a;
    z[3] = 0.;
    w[3] = *v;
    x[4] = 0.;
    y[4] = -(*a);
    z[4] = 0.;
    w[4] = *v;
    x[5] = 0.;
    y[5] = 0.;
    z[5] = *a;
    w[5] = *v;
    x[6] = 0.;
    y[6] = 0.;
    z[6] = -(*a);
    w[6] = *v;
    *num += 6;
    return 0;
/* vw */
L2:
    *a = sqrt(.5);
    x[1] = 0.;
    y[1] = *a;
    z[1] = *a;
    w[1] = *v;
    x[2] = 0.;
    y[2] = -(*a);
    z[2] = *a;
    w[2] = *v;
    x[3] = 0.;
    y[3] = *a;
    z[3] = -(*a);
    w[3] = *v;
    x[4] = 0.;
    y[4] = -(*a);
    z[4] = -(*a);
    w[4] = *v;
    x[5] = *a;
    y[5] = 0.;
    z[5] = *a;
    w[5] = *v;
    x[6] = -(*a);
    y[6] = 0.;
    z[6] = *a;
    w[6] = *v;
    x[7] = *a;
    y[7] = 0.;
    z[7] = -(*a);
    w[7] = *v;
    x[8] = -(*a);
    y[8] = 0.;
    z[8] = -(*a);
    w[8] = *v;
    x[9] = *a;
    y[9] = *a;
    z[9] = 0.;
    w[9] = *v;
    x[10] = -(*a);
    y[10] = *a;
    z[10] = 0.;
    w[10] = *v;
    x[11] = *a;
    y[11] = -(*a);
    z[11] = 0.;
    w[11] = *v;
    x[12] = -(*a);
    y[12] = -(*a);
    z[12] = 0.;
    w[12] = *v;
    *num += 12;
    return 0;
/* vw */
L3:
    *a = sqrt(.33333333333333331);
    x[1] = *a;
    y[1] = *a;
    z[1] = *a;
    w[1] = *v;
    x[2] = -(*a);
    y[2] = *a;
    z[2] = *a;
    w[2] = *v;
    x[3] = *a;
    y[3] = -(*a);
    z[3] = *a;
    w[3] = *v;
    x[4] = -(*a);
    y[4] = -(*a);
    z[4] = *a;
    w[4] = *v;
    x[5] = *a;
    y[5] = *a;
    z[5] = -(*a);
    w[5] = *v;
    x[6] = -(*a);
    y[6] = *a;
    z[6] = -(*a);
    w[6] = *v;
    x[7] = *a;
    y[7] = -(*a);
    z[7] = -(*a);
    w[7] = *v;
    x[8] = -(*a);
    y[8] = -(*a);
    z[8] = -(*a);
    w[8] = *v;
    *num += 8;
    return 0;
/* vw */
L4:
    *b = sqrt(1. - *a * 2. * *a);
    x[1] = *a;
    y[1] = *a;
    z[1] = *b;
    w[1] = *v;
    x[2] = -(*a);
    y[2] = *a;
    z[2] = *b;
    w[2] = *v;
    x[3] = *a;
    y[3] = -(*a);
    z[3] = *b;
    w[3] = *v;
    x[4] = -(*a);
    y[4] = -(*a);
    z[4] = *b;
    w[4] = *v;
    x[5] = *a;
    y[5] = *a;
    z[5] = -(*b);
    w[5] = *v;
    x[6] = -(*a);
    y[6] = *a;
    z[6] = -(*b);
    w[6] = *v;
    x[7] = *a;
    y[7] = -(*a);
    z[7] = -(*b);
    w[7] = *v;
    x[8] = -(*a);
    y[8] = -(*a);
    z[8] = -(*b);
    w[8] = *v;
    x[9] = *a;
    y[9] = *b;
    z[9] = *a;
    w[9] = *v;
    x[10] = -(*a);
    y[10] = *b;
    z[10] = *a;
    w[10] = *v;
    x[11] = *a;
    y[11] = -(*b);
    z[11] = *a;
    w[11] = *v;
    x[12] = -(*a);
    y[12] = -(*b);
    z[12] = *a;
    w[12] = *v;
    x[13] = *a;
    y[13] = *b;
    z[13] = -(*a);
    w[13] = *v;
    x[14] = -(*a);
    y[14] = *b;
    z[14] = -(*a);
    w[14] = *v;
    x[15] = *a;
    y[15] = -(*b);
    z[15] = -(*a);
    w[15] = *v;
    x[16] = -(*a);
    y[16] = -(*b);
    z[16] = -(*a);
    w[16] = *v;
    x[17] = *b;
    y[17] = *a;
    z[17] = *a;
    w[17] = *v;
    x[18] = -(*b);
    y[18] = *a;
    z[18] = *a;
    w[18] = *v;
    x[19] = *b;
    y[19] = -(*a);
    z[19] = *a;
    w[19] = *v;
    x[20] = -(*b);
    y[20] = -(*a);
    z[20] = *a;
    w[20] = *v;
    x[21] = *b;
    y[21] = *a;
    z[21] = -(*a);
    w[21] = *v;
    x[22] = -(*b);
    y[22] = *a;
    z[22] = -(*a);
    w[22] = *v;
    x[23] = *b;
    y[23] = -(*a);
    z[23] = -(*a);
    w[23] = *v;
    x[24] = -(*b);
    y[24] = -(*a);
    z[24] = -(*a);
    w[24] = *v;
    *num += 24;
    return 0;
/* vw */
L5:
    *b = sqrt(1. - *a * *a);
    x[1] = *a;
    y[1] = *b;
    z[1] = 0.;
    w[1] = *v;
    x[2] = -(*a);
    y[2] = *b;
    z[2] = 0.;
    w[2] = *v;
    x[3] = *a;
    y[3] = -(*b);
    z[3] = 0.;
    w[3] = *v;
    x[4] = -(*a);
    y[4] = -(*b);
    z[4] = 0.;
    w[4] = *v;
    x[5] = *b;
    y[5] = *a;
    z[5] = 0.;
    w[5] = *v;
    x[6] = -(*b);
    y[6] = *a;
    z[6] = 0.;
    w[6] = *v;
    x[7] = *b;
    y[7] = -(*a);
    z[7] = 0.;
    w[7] = *v;
    x[8] = -(*b);
    y[8] = -(*a);
    z[8] = 0.;
    w[8] = *v;
    x[9] = *a;
    y[9] = 0.;
    z[9] = *b;
    w[9] = *v;
    x[10] = -(*a);
    y[10] = 0.;
    z[10] = *b;
    w[10] = *v;
    x[11] = *a;
    y[11] = 0.;
    z[11] = -(*b);
    w[11] = *v;
    x[12] = -(*a);
    y[12] = 0.;
    z[12] = -(*b);
    w[12] = *v;
    x[13] = *b;
    y[13] = 0.;
    z[13] = *a;
    w[13] = *v;
    x[14] = -(*b);
    y[14] = 0.;
    z[14] = *a;
    w[14] = *v;
    x[15] = *b;
    y[15] = 0.;
    z[15] = -(*a);
    w[15] = *v;
    x[16] = -(*b);
    y[16] = 0.;
    z[16] = -(*a);
    w[16] = *v;
    x[17] = 0.;
    y[17] = *a;
    z[17] = *b;
    w[17] = *v;
    x[18] = 0.;
    y[18] = -(*a);
    z[18] = *b;
    w[18] = *v;
    x[19] = 0.;
    y[19] = *a;
    z[19] = -(*b);
    w[19] = *v;
    x[20] = 0.;
    y[20] = -(*a);
    z[20] = -(*b);
    w[20] = *v;
    x[21] = 0.;
    y[21] = *b;
    z[21] = *a;
    w[21] = *v;
    x[22] = 0.;
    y[22] = -(*b);
    z[22] = *a;
    w[22] = *v;
    x[23] = 0.;
    y[23] = *b;
    z[23] = -(*a);
    w[23] = *v;
    x[24] = 0.;
    y[24] = -(*b);
    z[24] = -(*a);
    w[24] = *v;
    *num += 24;
    return 0;
/* vw */
L6:
    c = sqrt(1. - *a * *a - *b * *b);
    x[1] = *a;
    y[1] = *b;
    z[1] = c;
    w[1] = *v;
    x[2] = -(*a);
    y[2] = *b;
    z[2] = c;
    w[2] = *v;
    x[3] = *a;
    y[3] = -(*b);
    z[3] = c;
    w[3] = *v;
    x[4] = -(*a);
    y[4] = -(*b);
    z[4] = c;
    w[4] = *v;
    x[5] = *a;
    y[5] = *b;
    z[5] = -c;
    w[5] = *v;
    x[6] = -(*a);
    y[6] = *b;
    z[6] = -c;
    w[6] = *v;
    x[7] = *a;
    y[7] = -(*b);
    z[7] = -c;
    w[7] = *v;
    x[8] = -(*a);
    y[8] = -(*b);
    z[8] = -c;
    w[8] = *v;
    x[9] = *a;
    y[9] = c;
    z[9] = *b;
    w[9] = *v;
    x[10] = -(*a);
    y[10] = c;
    z[10] = *b;
    w[10] = *v;
    x[11] = *a;
    y[11] = -c;
    z[11] = *b;
    w[11] = *v;
    x[12] = -(*a);
    y[12] = -c;
    z[12] = *b;
    w[12] = *v;
    x[13] = *a;
    y[13] = c;
    z[13] = -(*b);
    w[13] = *v;
    x[14] = -(*a);
    y[14] = c;
    z[14] = -(*b);
    w[14] = *v;
    x[15] = *a;
    y[15] = -c;
    z[15] = -(*b);
    w[15] = *v;
    x[16] = -(*a);
    y[16] = -c;
    z[16] = -(*b);
    w[16] = *v;
    x[17] = *b;
    y[17] = *a;
    z[17] = c;
    w[17] = *v;
    x[18] = -(*b);
    y[18] = *a;
    z[18] = c;
    w[18] = *v;
    x[19] = *b;
    y[19] = -(*a);
    z[19] = c;
    w[19] = *v;
    x[20] = -(*b);
    y[20] = -(*a);
    z[20] = c;
    w[20] = *v;
    x[21] = *b;
    y[21] = *a;
    z[21] = -c;
    w[21] = *v;
    x[22] = -(*b);
    y[22] = *a;
    z[22] = -c;
    w[22] = *v;
    x[23] = *b;
    y[23] = -(*a);
    z[23] = -c;
    w[23] = *v;
    x[24] = -(*b);
    y[24] = -(*a);
    z[24] = -c;
    w[24] = *v;
    x[25] = *b;
    y[25] = c;
    z[25] = *a;
    w[25] = *v;
    x[26] = -(*b);
    y[26] = c;
    z[26] = *a;
    w[26] = *v;
    x[27] = *b;
    y[27] = -c;
    z[27] = *a;
    w[27] = *v;
    x[28] = -(*b);
    y[28] = -c;
    z[28] = *a;
    w[28] = *v;
    x[29] = *b;
    y[29] = c;
    z[29] = -(*a);
    w[29] = *v;
    x[30] = -(*b);
    y[30] = c;
    z[30] = -(*a);
    w[30] = *v;
    x[31] = *b;
    y[31] = -c;
    z[31] = -(*a);
    w[31] = *v;
    x[32] = -(*b);
    y[32] = -c;
    z[32] = -(*a);
    w[32] = *v;
    x[33] = c;
    y[33] = *a;
    z[33] = *b;
    w[33] = *v;
    x[34] = -c;
    y[34] = *a;
    z[34] = *b;
    w[34] = *v;
    x[35] = c;
    y[35] = -(*a);
    z[35] = *b;
    w[35] = *v;
    x[36] = -c;
    y[36] = -(*a);
    z[36] = *b;
    w[36] = *v;
    x[37] = c;
    y[37] = *a;
    z[37] = -(*b);
    w[37] = *v;
    x[38] = -c;
    y[38] = *a;
    z[38] = -(*b);
    w[38] = *v;
    x[39] = c;
    y[39] = -(*a);
    z[39] = -(*b);
    w[39] = *v;
    x[40] = -c;
    y[40] = -(*a);
    z[40] = -(*b);
    w[40] = *v;
    x[41] = c;
    y[41] = *b;
    z[41] = *a;
    w[41] = *v;
    x[42] = -c;
    y[42] = *b;
    z[42] = *a;
    w[42] = *v;
    x[43] = c;
    y[43] = -(*b);
    z[43] = *a;
    w[43] = *v;
    x[44] = -c;
    y[44] = -(*b);
    z[44] = *a;
    w[44] = *v;
    x[45] = c;
    y[45] = *b;
    z[45] = -(*a);
    w[45] = *v;
    x[46] = -c;
    y[46] = *b;
    z[46] = -(*a);
    w[46] = *v;
    x[47] = c;
    y[47] = -(*b);
    z[47] = -(*a);
    w[47] = *v;
    x[48] = -c;
    y[48] = -(*b);
    z[48] = -(*a);
    w[48] = *v;
    *num += 48;
    return 0;
}

int ld0006_(double *x, double *y, double *z, double *w, int *n)
{
    static double a, b, v;

    /* VW */
    /* VW    LEBEDEV    6-POINT ANGULAR GRID */
    /* VW */
    /* hvd */
    /* hvd   This subroutine is part of a set of subroutines that generate */
    /* hvd   Lebedev grids [1-6] for integration on a sphere. The original */
    /* hvd   C-code [1] was kindly provided by Dr. Dmitri N. Laikov and */
    /* hvd   translated into fortran by Dr. Christoph van Wuellen. */
    /* hvd   This subroutine was translated using a C to fortran77 conversion */
    /* hvd   tool written by Dr. Christoph van Wuellen. */
    /* hvd */
    /* hvd   Users of this code are asked to include reference [1] in their */
    /* hvd   publications, and in the user- and programmers-manuals */
    /* hvd   describing their codes. */
    /* hvd */
    /* hvd   This code was distributed through CCL (http://www.ccl.net/). */
    /* hvd */
    /* hvd   [1] V.I. Lebedev, and D.N. Laikov */
    /* hvd       "A quadrature formula for the sphere of the 131st */
    /* hvd        algebraic order of accuracy" */
    /* hvd       Doklady Mathematics, Vol. 59, No. 3, 1999, pp. 477-481. */
    /* hvd */
    /* hvd   [2] V.I. Lebedev */
    /* hvd       "A quadrature formula for the sphere of 59th algebraic */
    /* hvd        order of accuracy" */
    /* hvd       Russian Acad. Sci. Dokl. Math., Vol. 50, 1995, pp. 283-286. */
    /* hvd */
    /* hvd   [3] V.I. Lebedev, and A.L. Skorokhodov */
    /* hvd       "Quadrature formulas of orders 41, 47, and 53 for the sphere" */
    /* hvd       Russian Acad. Sci. Dokl. Math., Vol. 45, 1992, pp. 587-592. */
    /* hvd */
    /* hvd   [4] V.I. Lebedev */
    /* hvd       "Spherical quadrature formulas exact to orders 25-29" */
    /* hvd       Siberian Mathematical Journal, Vol. 18, 1977, pp. 99-107. */
    /* hvd */
    /* hvd   [5] V.I. Lebedev */
    /* hvd       "Quadratures on a sphere" */
    /* hvd       Computational Mathematics and Mathematical Physics, Vol. 16, */
    /* hvd       1976, pp. 10-24. */
    /* hvd */
    /* hvd   [6] V.I. Lebedev */
    /* hvd       "Values of the nodes and weights of ninth to seventeenth */
    /* hvd        order Gauss-Markov quadrature formulae invariant under the */
    /* hvd        octahedron group with inversion" */
    /* hvd       Computational Mathematics and Mathematical Physics, Vol. 15, */
    /* hvd       1975, pp. 44-51. */
    /* hvd */
    /* Parameter adjustments */
    --w;
    --z;
    --y;
    --x;

    /* Function Body */
    *n = 1;
    v = .1666666666666667;
    gen_oh_(1, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    --(*n);
    return 0;
}

int ld0014_(double *x, double *y, double *z, double *w, int *n)
{
    static double a, b, v;

    /* VW */
    /* VW    LEBEDEV   14-POINT ANGULAR GRID */
    /* VW */
    /* hvd */
    /* hvd   This subroutine is part of a set of subroutines that generate */
    /* hvd   Lebedev grids [1-6] for integration on a sphere. The original */
    /* hvd   C-code [1] was kindly provided by Dr. Dmitri N. Laikov and */
    /* hvd   translated into fortran by Dr. Christoph van Wuellen. */
    /* hvd   This subroutine was translated using a C to fortran77 conversion */
    /* hvd   tool written by Dr. Christoph van Wuellen. */
    /* hvd */
    /* hvd   Users of this code are asked to include reference [1] in their */
    /* hvd   publications, and in the user- and programmers-manuals */
    /* hvd   describing their codes. */
    /* hvd */
    /* hvd   This code was distributed through CCL (http://www.ccl.net/). */
    /* hvd */
    /* hvd   [1] V.I. Lebedev, and D.N. Laikov */
    /* hvd       "A quadrature formula for the sphere of the 131st */
    /* hvd        algebraic order of accuracy" */
    /* hvd       Doklady Mathematics, Vol. 59, No. 3, 1999, pp. 477-481. */
    /* hvd */
    /* hvd   [2] V.I. Lebedev */
    /* hvd       "A quadrature formula for the sphere of 59th algebraic */
    /* hvd        order of accuracy" */
    /* hvd       Russian Acad. Sci. Dokl. Math., Vol. 50, 1995, pp. 283-286. */
    /* hvd */
    /* hvd   [3] V.I. Lebedev, and A.L. Skorokhodov */
    /* hvd       "Quadrature formulas of orders 41, 47, and 53 for the sphere" */
    /* hvd       Russian Acad. Sci. Dokl. Math., Vol. 45, 1992, pp. 587-592. */
    /* hvd */
    /* hvd   [4] V.I. Lebedev */
    /* hvd       "Spherical quadrature formulas exact to orders 25-29" */
    /* hvd       Siberian Mathematical Journal, Vol. 18, 1977, pp. 99-107. */
    /* hvd */
    /* hvd   [5] V.I. Lebedev */
    /* hvd       "Quadratures on a sphere" */
    /* hvd       Computational Mathematics and Mathematical Physics, Vol. 16, */
    /* hvd       1976, pp. 10-24. */
    /* hvd */
    /* hvd   [6] V.I. Lebedev */
    /* hvd       "Values of the nodes and weights of ninth to seventeenth */
    /* hvd        order Gauss-Markov quadrature formulae invariant under the */
    /* hvd        octahedron group with inversion" */
    /* hvd       Computational Mathematics and Mathematical Physics, Vol. 15, */
    /* hvd       1975, pp. 44-51. */
    /* hvd */
    /* Parameter adjustments */
    --w;
    --z;
    --y;
    --x;

    /* Function Body */
    *n = 1;
    v = .06666666666666667;
    gen_oh_(1, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    v = .075;
    gen_oh_(3, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    --(*n);
    return 0;
}

int ld0026_(double *x, double *y, double *z, double *w, int *n)
{
    static double a, b, v;

    /* VW */
    /* VW    LEBEDEV   26-POINT ANGULAR GRID */
    /* VW */
    /* hvd */
    /* hvd   This subroutine is part of a set of subroutines that generate */
    /* hvd   Lebedev grids [1-6] for integration on a sphere. The original */
    /* hvd   C-code [1] was kindly provided by Dr. Dmitri N. Laikov and */
    /* hvd   translated into fortran by Dr. Christoph van Wuellen. */
    /* hvd   This subroutine was translated using a C to fortran77 conversion */
    /* hvd   tool written by Dr. Christoph van Wuellen. */
    /* hvd */
    /* hvd   Users of this code are asked to include reference [1] in their */
    /* hvd   publications, and in the user- and programmers-manuals */
    /* hvd   describing their codes. */
    /* hvd */
    /* hvd   This code was distributed through CCL (http://www.ccl.net/). */
    /* hvd */
    /* hvd   [1] V.I. Lebedev, and D.N. Laikov */
    /* hvd       "A quadrature formula for the sphere of the 131st */
    /* hvd        algebraic order of accuracy" */
    /* hvd       Doklady Mathematics, Vol. 59, No. 3, 1999, pp. 477-481. */
    /* hvd */
    /* hvd   [2] V.I. Lebedev */
    /* hvd       "A quadrature formula for the sphere of 59th algebraic */
    /* hvd        order of accuracy" */
    /* hvd       Russian Acad. Sci. Dokl. Math., Vol. 50, 1995, pp. 283-286. */
    /* hvd */
    /* hvd   [3] V.I. Lebedev, and A.L. Skorokhodov */
    /* hvd       "Quadrature formulas of orders 41, 47, and 53 for the sphere" */
    /* hvd       Russian Acad. Sci. Dokl. Math., Vol. 45, 1992, pp. 587-592. */
    /* hvd */
    /* hvd   [4] V.I. Lebedev */
    /* hvd       "Spherical quadrature formulas exact to orders 25-29" */
    /* hvd       Siberian Mathematical Journal, Vol. 18, 1977, pp. 99-107. */
    /* hvd */
    /* hvd   [5] V.I. Lebedev */
    /* hvd       "Quadratures on a sphere" */
    /* hvd       Computational Mathematics and Mathematical Physics, Vol. 16, */
    /* hvd       1976, pp. 10-24. */
    /* hvd */
    /* hvd   [6] V.I. Lebedev */
    /* hvd       "Values of the nodes and weights of ninth to seventeenth */
    /* hvd        order Gauss-Markov quadrature formulae invariant under the */
    /* hvd        octahedron group with inversion" */
    /* hvd       Computational Mathematics and Mathematical Physics, Vol. 15, */
    /* hvd       1975, pp. 44-51. */
    /* hvd */
    /* Parameter adjustments */
    --w;
    --z;
    --y;
    --x;

    /* Function Body */
    *n = 1;
    v = .04761904761904762;
    gen_oh_(1, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    v = .0380952380952381;
    gen_oh_(2, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    v = .03214285714285714;
    gen_oh_(3, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    --(*n);
    return 0;
}

int ld0038_(double *x, double *y, double *z, double *w, int *n)
{
    static double a, b, v;

    /* VW */
    /* VW    LEBEDEV   38-POINT ANGULAR GRID */
    /* VW */
    /* hvd */
    /* hvd   This subroutine is part of a set of subroutines that generate */
    /* hvd   Lebedev grids [1-6] for integration on a sphere. The original */
    /* hvd   C-code [1] was kindly provided by Dr. Dmitri N. Laikov and */
    /* hvd   translated into fortran by Dr. Christoph van Wuellen. */
    /* hvd   This subroutine was translated using a C to fortran77 conversion */
    /* hvd   tool written by Dr. Christoph van Wuellen. */
    /* hvd */
    /* hvd   Users of this code are asked to include reference [1] in their */
    /* hvd   publications, and in the user- and programmers-manuals */
    /* hvd   describing their codes. */
    /* hvd */
    /* hvd   This code was distributed through CCL (http://www.ccl.net/). */
    /* hvd */
    /* hvd   [1] V.I. Lebedev, and D.N. Laikov */
    /* hvd       "A quadrature formula for the sphere of the 131st */
    /* hvd        algebraic order of accuracy" */
    /* hvd       Doklady Mathematics, Vol. 59, No. 3, 1999, pp. 477-481. */
    /* hvd */
    /* hvd   [2] V.I. Lebedev */
    /* hvd       "A quadrature formula for the sphere of 59th algebraic */
    /* hvd        order of accuracy" */
    /* hvd       Russian Acad. Sci. Dokl. Math., Vol. 50, 1995, pp. 283-286. */
    /* hvd */
    /* hvd   [3] V.I. Lebedev, and A.L. Skorokhodov */
    /* hvd       "Quadrature formulas of orders 41, 47, and 53 for the sphere" */
    /* hvd       Russian Acad. Sci. Dokl. Math., Vol. 45, 1992, pp. 587-592. */
    /* hvd */
    /* hvd   [4] V.I. Lebedev */
    /* hvd       "Spherical quadrature formulas exact to orders 25-29" */
    /* hvd       Siberian Mathematical Journal, Vol. 18, 1977, pp. 99-107. */
    /* hvd */
    /* hvd   [5] V.I. Lebedev */
    /* hvd       "Quadratures on a sphere" */
    /* hvd       Computational Mathematics and Mathematical Physics, Vol. 16, */
    /* hvd       1976, pp. 10-24. */
    /* hvd */
    /* hvd   [6] V.I. Lebedev */
    /* hvd       "Values of the nodes and weights of ninth to seventeenth */
    /* hvd        order Gauss-Markov quadrature formulae invariant under the */
    /* hvd        octahedron group with inversion" */
    /* hvd       Computational Mathematics and Mathematical Physics, Vol. 15, */
    /* hvd       1975, pp. 44-51. */
    /* hvd */
    /* Parameter adjustments */
    --w;
    --z;
    --y;
    --x;

    /* Function Body */
    *n = 1;
    v = .009523809523809524;
    gen_oh_(1, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    v = .03214285714285714;
    gen_oh_(3, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .4597008433809831;
    v = .02857142857142857;
    gen_oh_(5, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    --(*n);
    return 0;
}

int ld0050_(double *x, double *y, double *z, double *w, int *n)
{
    static double a, b, v;

    /* VW */
    /* VW    LEBEDEV   50-POINT ANGULAR GRID */
    /* VW */
    /* hvd */
    /* hvd   This subroutine is part of a set of subroutines that generate */
    /* hvd   Lebedev grids [1-6] for integration on a sphere. The original */
    /* hvd   C-code [1] was kindly provided by Dr. Dmitri N. Laikov and */
    /* hvd   translated into fortran by Dr. Christoph van Wuellen. */
    /* hvd   This subroutine was translated using a C to fortran77 conversion */
    /* hvd   tool written by Dr. Christoph van Wuellen. */
    /* hvd */
    /* hvd   Users of this code are asked to include reference [1] in their */
    /* hvd   publications, and in the user- and programmers-manuals */
    /* hvd   describing their codes. */
    /* hvd */
    /* hvd   This code was distributed through CCL (http://www.ccl.net/). */
    /* hvd */
    /* hvd   [1] V.I. Lebedev, and D.N. Laikov */
    /* hvd       "A quadrature formula for the sphere of the 131st */
    /* hvd        algebraic order of accuracy" */
    /* hvd       Doklady Mathematics, Vol. 59, No. 3, 1999, pp. 477-481. */
    /* hvd */
    /* hvd   [2] V.I. Lebedev */
    /* hvd       "A quadrature formula for the sphere of 59th algebraic */
    /* hvd        order of accuracy" */
    /* hvd       Russian Acad. Sci. Dokl. Math., Vol. 50, 1995, pp. 283-286. */
    /* hvd */
    /* hvd   [3] V.I. Lebedev, and A.L. Skorokhodov */
    /* hvd       "Quadrature formulas of orders 41, 47, and 53 for the sphere" */
    /* hvd       Russian Acad. Sci. Dokl. Math., Vol. 45, 1992, pp. 587-592. */
    /* hvd */
    /* hvd   [4] V.I. Lebedev */
    /* hvd       "Spherical quadrature formulas exact to orders 25-29" */
    /* hvd       Siberian Mathematical Journal, Vol. 18, 1977, pp. 99-107. */
    /* hvd */
    /* hvd   [5] V.I. Lebedev */
    /* hvd       "Quadratures on a sphere" */
    /* hvd       Computational Mathematics and Mathematical Physics, Vol. 16, */
    /* hvd       1976, pp. 10-24. */
    /* hvd */
    /* hvd   [6] V.I. Lebedev */
    /* hvd       "Values of the nodes and weights of ninth to seventeenth */
    /* hvd        order Gauss-Markov quadrature formulae invariant under the */
    /* hvd        octahedron group with inversion" */
    /* hvd       Computational Mathematics and Mathematical Physics, Vol. 15, */
    /* hvd       1975, pp. 44-51. */
    /* hvd */
    /* Parameter adjustments */
    --w;
    --z;
    --y;
    --x;

    /* Function Body */
    *n = 1;
    v = .0126984126984127;
    gen_oh_(1, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    v = .02257495590828924;
    gen_oh_(2, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    v = .02109375;
    gen_oh_(3, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .3015113445777636;
    v = .02017333553791887;
    gen_oh_(4, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    --(*n);
    return 0;
}

int ld0074_(double *x, double *y, double *z, double *w, int *n)
{
    static double a, b, v;

    /* VW */
    /* VW    LEBEDEV   74-POINT ANGULAR GRID */
    /* VW */
    /* hvd */
    /* hvd   This subroutine is part of a set of subroutines that generate */
    /* hvd   Lebedev grids [1-6] for integration on a sphere. The original */
    /* hvd   C-code [1] was kindly provided by Dr. Dmitri N. Laikov and */
    /* hvd   translated into fortran by Dr. Christoph van Wuellen. */
    /* hvd   This subroutine was translated using a C to fortran77 conversion */
    /* hvd   tool written by Dr. Christoph van Wuellen. */
    /* hvd */
    /* hvd   Users of this code are asked to include reference [1] in their */
    /* hvd   publications, and in the user- and programmers-manuals */
    /* hvd   describing their codes. */
    /* hvd */
    /* hvd   This code was distributed through CCL (http://www.ccl.net/). */
    /* hvd */
    /* hvd   [1] V.I. Lebedev, and D.N. Laikov */
    /* hvd       "A quadrature formula for the sphere of the 131st */
    /* hvd        algebraic order of accuracy" */
    /* hvd       Doklady Mathematics, Vol. 59, No. 3, 1999, pp. 477-481. */
    /* hvd */
    /* hvd   [2] V.I. Lebedev */
    /* hvd       "A quadrature formula for the sphere of 59th algebraic */
    /* hvd        order of accuracy" */
    /* hvd       Russian Acad. Sci. Dokl. Math., Vol. 50, 1995, pp. 283-286. */
    /* hvd */
    /* hvd   [3] V.I. Lebedev, and A.L. Skorokhodov */
    /* hvd       "Quadrature formulas of orders 41, 47, and 53 for the sphere" */
    /* hvd       Russian Acad. Sci. Dokl. Math., Vol. 45, 1992, pp. 587-592. */
    /* hvd */
    /* hvd   [4] V.I. Lebedev */
    /* hvd       "Spherical quadrature formulas exact to orders 25-29" */
    /* hvd       Siberian Mathematical Journal, Vol. 18, 1977, pp. 99-107. */
    /* hvd */
    /* hvd   [5] V.I. Lebedev */
    /* hvd       "Quadratures on a sphere" */
    /* hvd       Computational Mathematics and Mathematical Physics, Vol. 16, */
    /* hvd       1976, pp. 10-24. */
    /* hvd */
    /* hvd   [6] V.I. Lebedev */
    /* hvd       "Values of the nodes and weights of ninth to seventeenth */
    /* hvd        order Gauss-Markov quadrature formulae invariant under the */
    /* hvd        octahedron group with inversion" */
    /* hvd       Computational Mathematics and Mathematical Physics, Vol. 15, */
    /* hvd       1975, pp. 44-51. */
    /* hvd */
    /* Parameter adjustments */
    --w;
    --z;
    --y;
    --x;

    /* Function Body */
    *n = 1;
    v = 5.130671797338464e-4;
    gen_oh_(1, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    v = .01660406956574204;
    gen_oh_(2, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    v = -.02958603896103896;
    gen_oh_(3, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .4803844614152614;
    v = .02657620708215946;
    gen_oh_(4, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .3207726489807764;
    v = .01652217099371571;
    gen_oh_(5, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    --(*n);
    return 0;
}

int ld0086_(double *x, double *y, double *z, double *w, int *n)
{
    static double a, b, v;

    /* VW */
    /* VW    LEBEDEV   86-POINT ANGULAR GRID */
    /* VW */
    /* hvd */
    /* hvd   This subroutine is part of a set of subroutines that generate */
    /* hvd   Lebedev grids [1-6] for integration on a sphere. The original */
    /* hvd   C-code [1] was kindly provided by Dr. Dmitri N. Laikov and */
    /* hvd   translated into fortran by Dr. Christoph van Wuellen. */
    /* hvd   This subroutine was translated using a C to fortran77 conversion */
    /* hvd   tool written by Dr. Christoph van Wuellen. */
    /* hvd */
    /* hvd   Users of this code are asked to include reference [1] in their */
    /* hvd   publications, and in the user- and programmers-manuals */
    /* hvd   describing their codes. */
    /* hvd */
    /* hvd   This code was distributed through CCL (http://www.ccl.net/). */
    /* hvd */
    /* hvd   [1] V.I. Lebedev, and D.N. Laikov */
    /* hvd       "A quadrature formula for the sphere of the 131st */
    /* hvd        algebraic order of accuracy" */
    /* hvd       Doklady Mathematics, Vol. 59, No. 3, 1999, pp. 477-481. */
    /* hvd */
    /* hvd   [2] V.I. Lebedev */
    /* hvd       "A quadrature formula for the sphere of 59th algebraic */
    /* hvd        order of accuracy" */
    /* hvd       Russian Acad. Sci. Dokl. Math., Vol. 50, 1995, pp. 283-286. */
    /* hvd */
    /* hvd   [3] V.I. Lebedev, and A.L. Skorokhodov */
    /* hvd       "Quadrature formulas of orders 41, 47, and 53 for the sphere" */
    /* hvd       Russian Acad. Sci. Dokl. Math., Vol. 45, 1992, pp. 587-592. */
    /* hvd */
    /* hvd   [4] V.I. Lebedev */
    /* hvd       "Spherical quadrature formulas exact to orders 25-29" */
    /* hvd       Siberian Mathematical Journal, Vol. 18, 1977, pp. 99-107. */
    /* hvd */
    /* hvd   [5] V.I. Lebedev */
    /* hvd       "Quadratures on a sphere" */
    /* hvd       Computational Mathematics and Mathematical Physics, Vol. 16, */
    /* hvd       1976, pp. 10-24. */
    /* hvd */
    /* hvd   [6] V.I. Lebedev */
    /* hvd       "Values of the nodes and weights of ninth to seventeenth */
    /* hvd        order Gauss-Markov quadrature formulae invariant under the */
    /* hvd        octahedron group with inversion" */
    /* hvd       Computational Mathematics and Mathematical Physics, Vol. 15, */
    /* hvd       1975, pp. 44-51. */
    /* hvd */
    /* Parameter adjustments */
    --w;
    --z;
    --y;
    --x;

    /* Function Body */
    *n = 1;
    v = .01154401154401154;
    gen_oh_(1, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    v = .01194390908585628;
    gen_oh_(3, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .3696028464541502;
    v = .0111105557106034;
    gen_oh_(4, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .6943540066026664;
    v = .01187650129453714;
    gen_oh_(4, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .3742430390903412;
    v = .01181230374690448;
    gen_oh_(5, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    --(*n);
    return 0;
}

int ld0110_(double *x, double *y, double *z, double *w, int *n)
{
    static double a, b, v;

    /* VW */
    /* VW    LEBEDEV  110-POINT ANGULAR GRID */
    /* VW */
    /* hvd */
    /* hvd   This subroutine is part of a set of subroutines that generate */
    /* hvd   Lebedev grids [1-6] for integration on a sphere. The original */
    /* hvd   C-code [1] was kindly provided by Dr. Dmitri N. Laikov and */
    /* hvd   translated into fortran by Dr. Christoph van Wuellen. */
    /* hvd   This subroutine was translated using a C to fortran77 conversion */
    /* hvd   tool written by Dr. Christoph van Wuellen. */
    /* hvd */
    /* hvd   Users of this code are asked to include reference [1] in their */
    /* hvd   publications, and in the user- and programmers-manuals */
    /* hvd   describing their codes. */
    /* hvd */
    /* hvd   This code was distributed through CCL (http://www.ccl.net/). */
    /* hvd */
    /* hvd   [1] V.I. Lebedev, and D.N. Laikov */
    /* hvd       "A quadrature formula for the sphere of the 131st */
    /* hvd        algebraic order of accuracy" */
    /* hvd       Doklady Mathematics, Vol. 59, No. 3, 1999, pp. 477-481. */
    /* hvd */
    /* hvd   [2] V.I. Lebedev */
    /* hvd       "A quadrature formula for the sphere of 59th algebraic */
    /* hvd        order of accuracy" */
    /* hvd       Russian Acad. Sci. Dokl. Math., Vol. 50, 1995, pp. 283-286. */
    /* hvd */
    /* hvd   [3] V.I. Lebedev, and A.L. Skorokhodov */
    /* hvd       "Quadrature formulas of orders 41, 47, and 53 for the sphere" */
    /* hvd       Russian Acad. Sci. Dokl. Math., Vol. 45, 1992, pp. 587-592. */
    /* hvd */
    /* hvd   [4] V.I. Lebedev */
    /* hvd       "Spherical quadrature formulas exact to orders 25-29" */
    /* hvd       Siberian Mathematical Journal, Vol. 18, 1977, pp. 99-107. */
    /* hvd */
    /* hvd   [5] V.I. Lebedev */
    /* hvd       "Quadratures on a sphere" */
    /* hvd       Computational Mathematics and Mathematical Physics, Vol. 16, */
    /* hvd       1976, pp. 10-24. */
    /* hvd */
    /* hvd   [6] V.I. Lebedev */
    /* hvd       "Values of the nodes and weights of ninth to seventeenth */
    /* hvd        order Gauss-Markov quadrature formulae invariant under the */
    /* hvd        octahedron group with inversion" */
    /* hvd       Computational Mathematics and Mathematical Physics, Vol. 15, */
    /* hvd       1975, pp. 44-51. */
    /* hvd */
    /* Parameter adjustments */
    --w;
    --z;
    --y;
    --x;

    /* Function Body */
    *n = 1;
    v = .003828270494937162;
    gen_oh_(1, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    v = .009793737512487512;
    gen_oh_(3, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .1851156353447362;
    v = .008211737283191111;
    gen_oh_(4, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .6904210483822922;
    v = .009942814891178103;
    gen_oh_(4, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .3956894730559419;
    v = .009595471336070963;
    gen_oh_(4, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .4783690288121502;
    v = .009694996361663028;
    gen_oh_(5, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    --(*n);
    return 0;
}

int ld0146_(double *x, double *y, double *z, double *w, int *n)
{
    static double a, b, v;

    /* VW */
    /* VW    LEBEDEV  146-POINT ANGULAR GRID */
    /* VW */
    /* hvd */
    /* hvd   This subroutine is part of a set of subroutines that generate */
    /* hvd   Lebedev grids [1-6] for integration on a sphere. The original */
    /* hvd   C-code [1] was kindly provided by Dr. Dmitri N. Laikov and */
    /* hvd   translated into fortran by Dr. Christoph van Wuellen. */
    /* hvd   This subroutine was translated using a C to fortran77 conversion */
    /* hvd   tool written by Dr. Christoph van Wuellen. */
    /* hvd */
    /* hvd   Users of this code are asked to include reference [1] in their */
    /* hvd   publications, and in the user- and programmers-manuals */
    /* hvd   describing their codes. */
    /* hvd */
    /* hvd   This code was distributed through CCL (http://www.ccl.net/). */
    /* hvd */
    /* hvd   [1] V.I. Lebedev, and D.N. Laikov */
    /* hvd       "A quadrature formula for the sphere of the 131st */
    /* hvd        algebraic order of accuracy" */
    /* hvd       Doklady Mathematics, Vol. 59, No. 3, 1999, pp. 477-481. */
    /* hvd */
    /* hvd   [2] V.I. Lebedev */
    /* hvd       "A quadrature formula for the sphere of 59th algebraic */
    /* hvd        order of accuracy" */
    /* hvd       Russian Acad. Sci. Dokl. Math., Vol. 50, 1995, pp. 283-286. */
    /* hvd */
    /* hvd   [3] V.I. Lebedev, and A.L. Skorokhodov */
    /* hvd       "Quadrature formulas of orders 41, 47, and 53 for the sphere" */
    /* hvd       Russian Acad. Sci. Dokl. Math., Vol. 45, 1992, pp. 587-592. */
    /* hvd */
    /* hvd   [4] V.I. Lebedev */
    /* hvd       "Spherical quadrature formulas exact to orders 25-29" */
    /* hvd       Siberian Mathematical Journal, Vol. 18, 1977, pp. 99-107. */
    /* hvd */
    /* hvd   [5] V.I. Lebedev */
    /* hvd       "Quadratures on a sphere" */
    /* hvd       Computational Mathematics and Mathematical Physics, Vol. 16, */
    /* hvd       1976, pp. 10-24. */
    /* hvd */
    /* hvd   [6] V.I. Lebedev */
    /* hvd       "Values of the nodes and weights of ninth to seventeenth */
    /* hvd        order Gauss-Markov quadrature formulae invariant under the */
    /* hvd        octahedron group with inversion" */
    /* hvd       Computational Mathematics and Mathematical Physics, Vol. 15, */
    /* hvd       1975, pp. 44-51. */
    /* hvd */
    /* Parameter adjustments */
    --w;
    --z;
    --y;
    --x;

    /* Function Body */
    *n = 1;
    v = 5.996313688621381e-4;
    gen_oh_(1, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    v = .007372999718620756;
    gen_oh_(2, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    v = .007210515360144488;
    gen_oh_(3, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .6764410400114264;
    v = .007116355493117555;
    gen_oh_(4, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .4174961227965453;
    v = .006753829486314477;
    gen_oh_(4, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .1574676672039082;
    v = .007574394159054034;
    gen_oh_(4, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .1403553811713183;
    b = .4493328323269557;
    v = .006991087353303262;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    --(*n);
    return 0;
}

int ld0170_(double *x, double *y, double *z, double *w, int *n)
{
    static double a, b, v;

    /* VW */
    /* VW    LEBEDEV  170-POINT ANGULAR GRID */
    /* VW */
    /* hvd */
    /* hvd   This subroutine is part of a set of subroutines that generate */
    /* hvd   Lebedev grids [1-6] for integration on a sphere. The original */
    /* hvd   C-code [1] was kindly provided by Dr. Dmitri N. Laikov and */
    /* hvd   translated into fortran by Dr. Christoph van Wuellen. */
    /* hvd   This subroutine was translated using a C to fortran77 conversion */
    /* hvd   tool written by Dr. Christoph van Wuellen. */
    /* hvd */
    /* hvd   Users of this code are asked to include reference [1] in their */
    /* hvd   publications, and in the user- and programmers-manuals */
    /* hvd   describing their codes. */
    /* hvd */
    /* hvd   This code was distributed through CCL (http://www.ccl.net/). */
    /* hvd */
    /* hvd   [1] V.I. Lebedev, and D.N. Laikov */
    /* hvd       "A quadrature formula for the sphere of the 131st */
    /* hvd        algebraic order of accuracy" */
    /* hvd       Doklady Mathematics, Vol. 59, No. 3, 1999, pp. 477-481. */
    /* hvd */
    /* hvd   [2] V.I. Lebedev */
    /* hvd       "A quadrature formula for the sphere of 59th algebraic */
    /* hvd        order of accuracy" */
    /* hvd       Russian Acad. Sci. Dokl. Math., Vol. 50, 1995, pp. 283-286. */
    /* hvd */
    /* hvd   [3] V.I. Lebedev, and A.L. Skorokhodov */
    /* hvd       "Quadrature formulas of orders 41, 47, and 53 for the sphere" */
    /* hvd       Russian Acad. Sci. Dokl. Math., Vol. 45, 1992, pp. 587-592. */
    /* hvd */
    /* hvd   [4] V.I. Lebedev */
    /* hvd       "Spherical quadrature formulas exact to orders 25-29" */
    /* hvd       Siberian Mathematical Journal, Vol. 18, 1977, pp. 99-107. */
    /* hvd */
    /* hvd   [5] V.I. Lebedev */
    /* hvd       "Quadratures on a sphere" */
    /* hvd       Computational Mathematics and Mathematical Physics, Vol. 16, */
    /* hvd       1976, pp. 10-24. */
    /* hvd */
    /* hvd   [6] V.I. Lebedev */
    /* hvd       "Values of the nodes and weights of ninth to seventeenth */
    /* hvd        order Gauss-Markov quadrature formulae invariant under the */
    /* hvd        octahedron group with inversion" */
    /* hvd       Computational Mathematics and Mathematical Physics, Vol. 15, */
    /* hvd       1975, pp. 44-51. */
    /* hvd */
    /* Parameter adjustments */
    --w;
    --z;
    --y;
    --x;

    /* Function Body */
    *n = 1;
    v = .005544842902037365;
    gen_oh_(1, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    v = .006071332770670752;
    gen_oh_(2, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    v = .006383674773515093;
    gen_oh_(3, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .2551252621114134;
    v = .00518338758774779;
    gen_oh_(4, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .6743601460362766;
    v = .006317929009813725;
    gen_oh_(4, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .431891069671941;
    v = .006201670006589077;
    gen_oh_(4, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .2613931360335988;
    v = .005477143385137348;
    gen_oh_(5, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .4990453161796037;
    b = .1446630744325115;
    v = .005968383987681156;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    --(*n);
    return 0;
}

int ld0194_(double *x, double *y, double *z, double *w, int *n)
{
    static double a, b, v;

    /* VW */
    /* VW    LEBEDEV  194-POINT ANGULAR GRID */
    /* VW */
    /* hvd */
    /* hvd   This subroutine is part of a set of subroutines that generate */
    /* hvd   Lebedev grids [1-6] for integration on a sphere. The original */
    /* hvd   C-code [1] was kindly provided by Dr. Dmitri N. Laikov and */
    /* hvd   translated into fortran by Dr. Christoph van Wuellen. */
    /* hvd   This subroutine was translated using a C to fortran77 conversion */
    /* hvd   tool written by Dr. Christoph van Wuellen. */
    /* hvd */
    /* hvd   Users of this code are asked to include reference [1] in their */
    /* hvd   publications, and in the user- and programmers-manuals */
    /* hvd   describing their codes. */
    /* hvd */
    /* hvd   This code was distributed through CCL (http://www.ccl.net/). */
    /* hvd */
    /* hvd   [1] V.I. Lebedev, and D.N. Laikov */
    /* hvd       "A quadrature formula for the sphere of the 131st */
    /* hvd        algebraic order of accuracy" */
    /* hvd       Doklady Mathematics, Vol. 59, No. 3, 1999, pp. 477-481. */
    /* hvd */
    /* hvd   [2] V.I. Lebedev */
    /* hvd       "A quadrature formula for the sphere of 59th algebraic */
    /* hvd        order of accuracy" */
    /* hvd       Russian Acad. Sci. Dokl. Math., Vol. 50, 1995, pp. 283-286. */
    /* hvd */
    /* hvd   [3] V.I. Lebedev, and A.L. Skorokhodov */
    /* hvd       "Quadrature formulas of orders 41, 47, and 53 for the sphere" */
    /* hvd       Russian Acad. Sci. Dokl. Math., Vol. 45, 1992, pp. 587-592. */
    /* hvd */
    /* hvd   [4] V.I. Lebedev */
    /* hvd       "Spherical quadrature formulas exact to orders 25-29" */
    /* hvd       Siberian Mathematical Journal, Vol. 18, 1977, pp. 99-107. */
    /* hvd */
    /* hvd   [5] V.I. Lebedev */
    /* hvd       "Quadratures on a sphere" */
    /* hvd       Computational Mathematics and Mathematical Physics, Vol. 16, */
    /* hvd       1976, pp. 10-24. */
    /* hvd */
    /* hvd   [6] V.I. Lebedev */
    /* hvd       "Values of the nodes and weights of ninth to seventeenth */
    /* hvd        order Gauss-Markov quadrature formulae invariant under the */
    /* hvd        octahedron group with inversion" */
    /* hvd       Computational Mathematics and Mathematical Physics, Vol. 15, */
    /* hvd       1975, pp. 44-51. */
    /* hvd */
    /* Parameter adjustments */
    --w;
    --z;
    --y;
    --x;

    /* Function Body */
    *n = 1;
    v = .001782340447244611;
    gen_oh_(1, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    v = .005716905949977102;
    gen_oh_(2, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    v = .005573383178848738;
    gen_oh_(3, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .6712973442695226;
    v = .005608704082587997;
    gen_oh_(4, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .2892465627575439;
    v = .005158237711805383;
    gen_oh_(4, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .4446933178717437;
    v = .005518771467273614;
    gen_oh_(4, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .1299335447650067;
    v = .004106777028169394;
    gen_oh_(4, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .3457702197611283;
    v = .005051846064614808;
    gen_oh_(5, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .159041710538353;
    b = .8360360154824589;
    v = .005530248916233094;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    --(*n);
    return 0;
}

int ld0230_(double *x, double *y, double *z, double *w, int *n)
{
    static double a, b, v;

    /* VW */
    /* VW    LEBEDEV  230-POINT ANGULAR GRID */
    /* VW */
    /* hvd */
    /* hvd   This subroutine is part of a set of subroutines that generate */
    /* hvd   Lebedev grids [1-6] for integration on a sphere. The original */
    /* hvd   C-code [1] was kindly provided by Dr. Dmitri N. Laikov and */
    /* hvd   translated into fortran by Dr. Christoph van Wuellen. */
    /* hvd   This subroutine was translated using a C to fortran77 conversion */
    /* hvd   tool written by Dr. Christoph van Wuellen. */
    /* hvd */
    /* hvd   Users of this code are asked to include reference [1] in their */
    /* hvd   publications, and in the user- and programmers-manuals */
    /* hvd   describing their codes. */
    /* hvd */
    /* hvd   This code was distributed through CCL (http://www.ccl.net/). */
    /* hvd */
    /* hvd   [1] V.I. Lebedev, and D.N. Laikov */
    /* hvd       "A quadrature formula for the sphere of the 131st */
    /* hvd        algebraic order of accuracy" */
    /* hvd       Doklady Mathematics, Vol. 59, No. 3, 1999, pp. 477-481. */
    /* hvd */
    /* hvd   [2] V.I. Lebedev */
    /* hvd       "A quadrature formula for the sphere of 59th algebraic */
    /* hvd        order of accuracy" */
    /* hvd       Russian Acad. Sci. Dokl. Math., Vol. 50, 1995, pp. 283-286. */
    /* hvd */
    /* hvd   [3] V.I. Lebedev, and A.L. Skorokhodov */
    /* hvd       "Quadrature formulas of orders 41, 47, and 53 for the sphere" */
    /* hvd       Russian Acad. Sci. Dokl. Math., Vol. 45, 1992, pp. 587-592. */
    /* hvd */
    /* hvd   [4] V.I. Lebedev */
    /* hvd       "Spherical quadrature formulas exact to orders 25-29" */
    /* hvd       Siberian Mathematical Journal, Vol. 18, 1977, pp. 99-107. */
    /* hvd */
    /* hvd   [5] V.I. Lebedev */
    /* hvd       "Quadratures on a sphere" */
    /* hvd       Computational Mathematics and Mathematical Physics, Vol. 16, */
    /* hvd       1976, pp. 10-24. */
    /* hvd */
    /* hvd   [6] V.I. Lebedev */
    /* hvd       "Values of the nodes and weights of ninth to seventeenth */
    /* hvd        order Gauss-Markov quadrature formulae invariant under the */
    /* hvd        octahedron group with inversion" */
    /* hvd       Computational Mathematics and Mathematical Physics, Vol. 15, */
    /* hvd       1975, pp. 44-51. */
    /* hvd */
    /* Parameter adjustments */
    --w;
    --z;
    --y;
    --x;

    /* Function Body */
    *n = 1;
    v = -.05522639919727325;
    gen_oh_(1, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    v = .004450274607445226;
    gen_oh_(3, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .4492044687397611;
    v = .004496841067921404;
    gen_oh_(4, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .2520419490210201;
    v = .00504915345047875;
    gen_oh_(4, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .6981906658447242;
    v = .003976408018051883;
    gen_oh_(4, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .658740524346096;
    v = .004401400650381014;
    gen_oh_(4, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .0403854405009766;
    v = .01724544350544401;
    gen_oh_(4, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .5823842309715585;
    v = .004231083095357343;
    gen_oh_(5, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .3545877390518688;
    v = .005198069864064399;
    gen_oh_(5, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .2272181808998187;
    b = .4864661535886647;
    v = .004695720972568883;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    --(*n);
    return 0;
}

int ld0266_(double *x, double *y, double *z, double *w, int *n)
{
    static double a, b, v;

    /* VW */
    /* VW    LEBEDEV  266-POINT ANGULAR GRID */
    /* VW */
    /* hvd */
    /* hvd   This subroutine is part of a set of subroutines that generate */
    /* hvd   Lebedev grids [1-6] for integration on a sphere. The original */
    /* hvd   C-code [1] was kindly provided by Dr. Dmitri N. Laikov and */
    /* hvd   translated into fortran by Dr. Christoph van Wuellen. */
    /* hvd   This subroutine was translated using a C to fortran77 conversion */
    /* hvd   tool written by Dr. Christoph van Wuellen. */
    /* hvd */
    /* hvd   Users of this code are asked to include reference [1] in their */
    /* hvd   publications, and in the user- and programmers-manuals */
    /* hvd   describing their codes. */
    /* hvd */
    /* hvd   This code was distributed through CCL (http://www.ccl.net/). */
    /* hvd */
    /* hvd   [1] V.I. Lebedev, and D.N. Laikov */
    /* hvd       "A quadrature formula for the sphere of the 131st */
    /* hvd        algebraic order of accuracy" */
    /* hvd       Doklady Mathematics, Vol. 59, No. 3, 1999, pp. 477-481. */
    /* hvd */
    /* hvd   [2] V.I. Lebedev */
    /* hvd       "A quadrature formula for the sphere of 59th algebraic */
    /* hvd        order of accuracy" */
    /* hvd       Russian Acad. Sci. Dokl. Math., Vol. 50, 1995, pp. 283-286. */
    /* hvd */
    /* hvd   [3] V.I. Lebedev, and A.L. Skorokhodov */
    /* hvd       "Quadrature formulas of orders 41, 47, and 53 for the sphere" */
    /* hvd       Russian Acad. Sci. Dokl. Math., Vol. 45, 1992, pp. 587-592. */
    /* hvd */
    /* hvd   [4] V.I. Lebedev */
    /* hvd       "Spherical quadrature formulas exact to orders 25-29" */
    /* hvd       Siberian Mathematical Journal, Vol. 18, 1977, pp. 99-107. */
    /* hvd */
    /* hvd   [5] V.I. Lebedev */
    /* hvd       "Quadratures on a sphere" */
    /* hvd       Computational Mathematics and Mathematical Physics, Vol. 16, */
    /* hvd       1976, pp. 10-24. */
    /* hvd */
    /* hvd   [6] V.I. Lebedev */
    /* hvd       "Values of the nodes and weights of ninth to seventeenth */
    /* hvd        order Gauss-Markov quadrature formulae invariant under the */
    /* hvd        octahedron group with inversion" */
    /* hvd       Computational Mathematics and Mathematical Physics, Vol. 15, */
    /* hvd       1975, pp. 44-51. */
    /* hvd */
    /* Parameter adjustments */
    --w;
    --z;
    --y;
    --x;

    /* Function Body */
    *n = 1;
    v = -.001313769127326952;
    gen_oh_(1, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    v = -.002522728704859336;
    gen_oh_(2, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    v = .004186853881700583;
    gen_oh_(3, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .7039373391585475;
    v = .005315167977810885;
    gen_oh_(4, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .1012526248572414;
    v = .004047142377086219;
    gen_oh_(4, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .4647448726420539;
    v = .00411248239440699;
    gen_oh_(4, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .3277420654971629;
    v = .003595584899758782;
    gen_oh_(4, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .6620338663699974;
    v = .004256131351428158;
    gen_oh_(4, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .8506508083520399;
    v = .00422958270064724;
    gen_oh_(5, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .3233484542692899;
    b = .1153112011009701;
    v = .004080914225780505;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .2314790158712601;
    b = .5244939240922365;
    v = .004071467593830964;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    --(*n);
    return 0;
}

int ld0302_(double *x, double *y, double *z, double *w, int *n)
{
    static double a, b, v;

    /* VW */
    /* VW    LEBEDEV  302-POINT ANGULAR GRID */
    /* VW */
    /* hvd */
    /* hvd   This subroutine is part of a set of subroutines that generate */
    /* hvd   Lebedev grids [1-6] for integration on a sphere. The original */
    /* hvd   C-code [1] was kindly provided by Dr. Dmitri N. Laikov and */
    /* hvd   translated into fortran by Dr. Christoph van Wuellen. */
    /* hvd   This subroutine was translated using a C to fortran77 conversion */
    /* hvd   tool written by Dr. Christoph van Wuellen. */
    /* hvd */
    /* hvd   Users of this code are asked to include reference [1] in their */
    /* hvd   publications, and in the user- and programmers-manuals */
    /* hvd   describing their codes. */
    /* hvd */
    /* hvd   This code was distributed through CCL (http://www.ccl.net/). */
    /* hvd */
    /* hvd   [1] V.I. Lebedev, and D.N. Laikov */
    /* hvd       "A quadrature formula for the sphere of the 131st */
    /* hvd        algebraic order of accuracy" */
    /* hvd       Doklady Mathematics, Vol. 59, No. 3, 1999, pp. 477-481. */
    /* hvd */
    /* hvd   [2] V.I. Lebedev */
    /* hvd       "A quadrature formula for the sphere of 59th algebraic */
    /* hvd        order of accuracy" */
    /* hvd       Russian Acad. Sci. Dokl. Math., Vol. 50, 1995, pp. 283-286. */
    /* hvd */
    /* hvd   [3] V.I. Lebedev, and A.L. Skorokhodov */
    /* hvd       "Quadrature formulas of orders 41, 47, and 53 for the sphere" */
    /* hvd       Russian Acad. Sci. Dokl. Math., Vol. 45, 1992, pp. 587-592. */
    /* hvd */
    /* hvd   [4] V.I. Lebedev */
    /* hvd       "Spherical quadrature formulas exact to orders 25-29" */
    /* hvd       Siberian Mathematical Journal, Vol. 18, 1977, pp. 99-107. */
    /* hvd */
    /* hvd   [5] V.I. Lebedev */
    /* hvd       "Quadratures on a sphere" */
    /* hvd       Computational Mathematics and Mathematical Physics, Vol. 16, */
    /* hvd       1976, pp. 10-24. */
    /* hvd */
    /* hvd   [6] V.I. Lebedev */
    /* hvd       "Values of the nodes and weights of ninth to seventeenth */
    /* hvd        order Gauss-Markov quadrature formulae invariant under the */
    /* hvd        octahedron group with inversion" */
    /* hvd       Computational Mathematics and Mathematical Physics, Vol. 15, */
    /* hvd       1975, pp. 44-51. */
    /* hvd */
    /* Parameter adjustments */
    --w;
    --z;
    --y;
    --x;

    /* Function Body */
    *n = 1;
    v = 8.545911725128148e-4;
    gen_oh_(1, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    v = .003599119285025571;
    gen_oh_(3, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .3515640345570105;
    v = .003449788424305883;
    gen_oh_(4, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .6566329410219612;
    v = .003604822601419882;
    gen_oh_(4, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .4729054132581005;
    v = .003576729661743367;
    gen_oh_(4, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .09618308522614784;
    v = .002352101413689164;
    gen_oh_(4, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .2219645236294178;
    v = .003108953122413675;
    gen_oh_(4, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .7011766416089545;
    v = .003650045807677255;
    gen_oh_(4, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .2644152887060663;
    v = .002982344963171804;
    gen_oh_(5, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .5718955891878961;
    v = .00360082093221646;
    gen_oh_(5, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .2510034751770465;
    b = .8000727494073952;
    v = .003571540554273387;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .1233548532583327;
    b = .4127724083168531;
    v = .00339231220500617;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    --(*n);
    return 0;
}

int ld0350_(double *x, double *y, double *z, double *w, int *n)
{
    static double a, b, v;

    /* VW */
    /* VW    LEBEDEV  350-POINT ANGULAR GRID */
    /* VW */
    /* hvd */
    /* hvd   This subroutine is part of a set of subroutines that generate */
    /* hvd   Lebedev grids [1-6] for integration on a sphere. The original */
    /* hvd   C-code [1] was kindly provided by Dr. Dmitri N. Laikov and */
    /* hvd   translated into fortran by Dr. Christoph van Wuellen. */
    /* hvd   This subroutine was translated using a C to fortran77 conversion */
    /* hvd   tool written by Dr. Christoph van Wuellen. */
    /* hvd */
    /* hvd   Users of this code are asked to include reference [1] in their */
    /* hvd   publications, and in the user- and programmers-manuals */
    /* hvd   describing their codes. */
    /* hvd */
    /* hvd   This code was distributed through CCL (http://www.ccl.net/). */
    /* hvd */
    /* hvd   [1] V.I. Lebedev, and D.N. Laikov */
    /* hvd       "A quadrature formula for the sphere of the 131st */
    /* hvd        algebraic order of accuracy" */
    /* hvd       Doklady Mathematics, Vol. 59, No. 3, 1999, pp. 477-481. */
    /* hvd */
    /* hvd   [2] V.I. Lebedev */
    /* hvd       "A quadrature formula for the sphere of 59th algebraic */
    /* hvd        order of accuracy" */
    /* hvd       Russian Acad. Sci. Dokl. Math., Vol. 50, 1995, pp. 283-286. */
    /* hvd */
    /* hvd   [3] V.I. Lebedev, and A.L. Skorokhodov */
    /* hvd       "Quadrature formulas of orders 41, 47, and 53 for the sphere" */
    /* hvd       Russian Acad. Sci. Dokl. Math., Vol. 45, 1992, pp. 587-592. */
    /* hvd */
    /* hvd   [4] V.I. Lebedev */
    /* hvd       "Spherical quadrature formulas exact to orders 25-29" */
    /* hvd       Siberian Mathematical Journal, Vol. 18, 1977, pp. 99-107. */
    /* hvd */
    /* hvd   [5] V.I. Lebedev */
    /* hvd       "Quadratures on a sphere" */
    /* hvd       Computational Mathematics and Mathematical Physics, Vol. 16, */
    /* hvd       1976, pp. 10-24. */
    /* hvd */
    /* hvd   [6] V.I. Lebedev */
    /* hvd       "Values of the nodes and weights of ninth to seventeenth */
    /* hvd        order Gauss-Markov quadrature formulae invariant under the */
    /* hvd        octahedron group with inversion" */
    /* hvd       Computational Mathematics and Mathematical Physics, Vol. 15, */
    /* hvd       1975, pp. 44-51. */
    /* hvd */
    /* Parameter adjustments */
    --w;
    --z;
    --y;
    --x;

    /* Function Body */
    *n = 1;
    v = .003006796749453936;
    gen_oh_(1, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    v = .003050627745650771;
    gen_oh_(3, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .7068965463912316;
    v = .001621104600288991;
    gen_oh_(4, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .4794682625712025;
    v = .003005701484901752;
    gen_oh_(4, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .1927533154878019;
    v = .002990992529653774;
    gen_oh_(4, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .6930357961327123;
    v = .002982170644107595;
    gen_oh_(4, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .3608302115520091;
    v = .002721564237310992;
    gen_oh_(4, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .6498486161496169;
    v = .003033513795811141;
    gen_oh_(4, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .1932945013230339;
    v = .003007949555218533;
    gen_oh_(5, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .3800494919899303;
    v = .002881964603055307;
    gen_oh_(5, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .2899558825499574;
    b = .7934537856582316;
    v = .002958357626535696;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .09684121455103957;
    b = .8280801506686862;
    v = .003036020026407088;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .1833434647041659;
    b = .9074658265305127;
    v = .002832187403926303;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    --(*n);
    return 0;
}

int ld0434_(double *x, double *y, double *z, double *w, int *n)
{
    static double a, b, v;

    /* VW */
    /* VW    LEBEDEV  434-POINT ANGULAR GRID */
    /* VW */
    /* hvd */
    /* hvd   This subroutine is part of a set of subroutines that generate */
    /* hvd   Lebedev grids [1-6] for integration on a sphere. The original */
    /* hvd   C-code [1] was kindly provided by Dr. Dmitri N. Laikov and */
    /* hvd   translated into fortran by Dr. Christoph van Wuellen. */
    /* hvd   This subroutine was translated using a C to fortran77 conversion */
    /* hvd   tool written by Dr. Christoph van Wuellen. */
    /* hvd */
    /* hvd   Users of this code are asked to include reference [1] in their */
    /* hvd   publications, and in the user- and programmers-manuals */
    /* hvd   describing their codes. */
    /* hvd */
    /* hvd   This code was distributed through CCL (http://www.ccl.net/). */
    /* hvd */
    /* hvd   [1] V.I. Lebedev, and D.N. Laikov */
    /* hvd       "A quadrature formula for the sphere of the 131st */
    /* hvd        algebraic order of accuracy" */
    /* hvd       Doklady Mathematics, Vol. 59, No. 3, 1999, pp. 477-481. */
    /* hvd */
    /* hvd   [2] V.I. Lebedev */
    /* hvd       "A quadrature formula for the sphere of 59th algebraic */
    /* hvd        order of accuracy" */
    /* hvd       Russian Acad. Sci. Dokl. Math., Vol. 50, 1995, pp. 283-286. */
    /* hvd */
    /* hvd   [3] V.I. Lebedev, and A.L. Skorokhodov */
    /* hvd       "Quadrature formulas of orders 41, 47, and 53 for the sphere" */
    /* hvd       Russian Acad. Sci. Dokl. Math., Vol. 45, 1992, pp. 587-592. */
    /* hvd */
    /* hvd   [4] V.I. Lebedev */
    /* hvd       "Spherical quadrature formulas exact to orders 25-29" */
    /* hvd       Siberian Mathematical Journal, Vol. 18, 1977, pp. 99-107. */
    /* hvd */
    /* hvd   [5] V.I. Lebedev */
    /* hvd       "Quadratures on a sphere" */
    /* hvd       Computational Mathematics and Mathematical Physics, Vol. 16, */
    /* hvd       1976, pp. 10-24. */
    /* hvd */
    /* hvd   [6] V.I. Lebedev */
    /* hvd       "Values of the nodes and weights of ninth to seventeenth */
    /* hvd        order Gauss-Markov quadrature formulae invariant under the */
    /* hvd        octahedron group with inversion" */
    /* hvd       Computational Mathematics and Mathematical Physics, Vol. 15, */
    /* hvd       1975, pp. 44-51. */
    /* hvd */
    /* Parameter adjustments */
    --w;
    --z;
    --y;
    --x;

    /* Function Body */
    *n = 1;
    v = 5.265897968224436e-4;
    gen_oh_(1, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    v = .002548219972002607;
    gen_oh_(2, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    v = .002512317418927307;
    gen_oh_(3, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .6909346307509111;
    v = .002530403801186355;
    gen_oh_(4, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .1774836054609158;
    v = .002014279020918528;
    gen_oh_(4, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .4914342637784746;
    v = .002501725168402936;
    gen_oh_(4, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .6456664707424256;
    v = .002513267174597564;
    gen_oh_(4, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .2861289010307638;
    v = .002302694782227416;
    gen_oh_(4, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .07568084367178018;
    v = .001462495621594614;
    gen_oh_(4, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .3927259763368002;
    v = .00244537343731298;
    gen_oh_(4, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .8818132877794288;
    v = .002417442375638981;
    gen_oh_(5, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .9776428111182649;
    v = .001910951282179532;
    gen_oh_(5, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .2054823696403044;
    b = .8689460322872412;
    v = .002416930044324775;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .5905157048925271;
    b = .7999278543857286;
    v = .002512236854563495;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .5550152361076807;
    b = .7717462626915901;
    v = .002496644054553086;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .9371809858553722;
    b = .3344363145343455;
    v = .002236607760437849;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    --(*n);
    return 0;
}

int ld0590_(double *x, double *y, double *z, double *w, int *n)
{
    static double a, b, v;

    /* VW */
    /* VW    LEBEDEV  590-POINT ANGULAR GRID */
    /* VW */
    /* hvd */
    /* hvd   This subroutine is part of a set of subroutines that generate */
    /* hvd   Lebedev grids [1-6] for integration on a sphere. The original */
    /* hvd   C-code [1] was kindly provided by Dr. Dmitri N. Laikov and */
    /* hvd   translated into fortran by Dr. Christoph van Wuellen. */
    /* hvd   This subroutine was translated using a C to fortran77 conversion */
    /* hvd   tool written by Dr. Christoph van Wuellen. */
    /* hvd */
    /* hvd   Users of this code are asked to include reference [1] in their */
    /* hvd   publications, and in the user- and programmers-manuals */
    /* hvd   describing their codes. */
    /* hvd */
    /* hvd   This code was distributed through CCL (http://www.ccl.net/). */
    /* hvd */
    /* hvd   [1] V.I. Lebedev, and D.N. Laikov */
    /* hvd       "A quadrature formula for the sphere of the 131st */
    /* hvd        algebraic order of accuracy" */
    /* hvd       Doklady Mathematics, Vol. 59, No. 3, 1999, pp. 477-481. */
    /* hvd */
    /* hvd   [2] V.I. Lebedev */
    /* hvd       "A quadrature formula for the sphere of 59th algebraic */
    /* hvd        order of accuracy" */
    /* hvd       Russian Acad. Sci. Dokl. Math., Vol. 50, 1995, pp. 283-286. */
    /* hvd */
    /* hvd   [3] V.I. Lebedev, and A.L. Skorokhodov */
    /* hvd       "Quadrature formulas of orders 41, 47, and 53 for the sphere" */
    /* hvd       Russian Acad. Sci. Dokl. Math., Vol. 45, 1992, pp. 587-592. */
    /* hvd */
    /* hvd   [4] V.I. Lebedev */
    /* hvd       "Spherical quadrature formulas exact to orders 25-29" */
    /* hvd       Siberian Mathematical Journal, Vol. 18, 1977, pp. 99-107. */
    /* hvd */
    /* hvd   [5] V.I. Lebedev */
    /* hvd       "Quadratures on a sphere" */
    /* hvd       Computational Mathematics and Mathematical Physics, Vol. 16, */
    /* hvd       1976, pp. 10-24. */
    /* hvd */
    /* hvd   [6] V.I. Lebedev */
    /* hvd       "Values of the nodes and weights of ninth to seventeenth */
    /* hvd        order Gauss-Markov quadrature formulae invariant under the */
    /* hvd        octahedron group with inversion" */
    /* hvd       Computational Mathematics and Mathematical Physics, Vol. 15, */
    /* hvd       1975, pp. 44-51. */
    /* hvd */
    /* Parameter adjustments */
    --w;
    --z;
    --y;
    --x;

    /* Function Body */
    *n = 1;
    v = 3.095121295306187e-4;
    gen_oh_(1, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    v = .001852379698597489;
    gen_oh_(3, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .7040954938227469;
    v = .001871790639277744;
    gen_oh_(4, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .6807744066455243;
    v = .001858812585438317;
    gen_oh_(4, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .6372546939258752;
    v = .001852028828296213;
    gen_oh_(4, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .5044419707800358;
    v = .001846715956151242;
    gen_oh_(4, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .4215761784010967;
    v = .001818471778162769;
    gen_oh_(4, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .3317920736472123;
    v = .001749564657281154;
    gen_oh_(4, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .2384736701421887;
    v = .001617210647254411;
    gen_oh_(4, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .1459036449157763;
    v = .001384737234851692;
    gen_oh_(4, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .06095034115507196;
    v = 9.76433116505105e-4;
    gen_oh_(4, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .6116843442009876;
    v = .001857161196774078;
    gen_oh_(5, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .3964755348199858;
    v = .001705153996395864;
    gen_oh_(5, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .1724782009907724;
    v = .001300321685886048;
    gen_oh_(5, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .561026380862206;
    b = .3518280927733519;
    v = .001842866472905286;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .474239284255198;
    b = .263471665593795;
    v = .001802658934377451;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .598412649788538;
    b = .1816640840360209;
    v = .00184983056044366;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .3791035407695563;
    b = .1720795225656878;
    v = .001713904507106709;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .2778673190586244;
    b = .08213021581932511;
    v = .001555213603396808;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .5033564271075117;
    b = .08999205842074875;
    v = .001802239128008525;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    --(*n);
    return 0;
}

int ld0770_(double *x, double *y, double *z, double *w, int *n)
{
    static double a, b, v;

    /* VW */
    /* VW    LEBEDEV  770-POINT ANGULAR GRID */
    /* VW */
    /* hvd */
    /* hvd   This subroutine is part of a set of subroutines that generate */
    /* hvd   Lebedev grids [1-6] for integration on a sphere. The original */
    /* hvd   C-code [1] was kindly provided by Dr. Dmitri N. Laikov and */
    /* hvd   translated into fortran by Dr. Christoph van Wuellen. */
    /* hvd   This subroutine was translated using a C to fortran77 conversion */
    /* hvd   tool written by Dr. Christoph van Wuellen. */
    /* hvd */
    /* hvd   Users of this code are asked to include reference [1] in their */
    /* hvd   publications, and in the user- and programmers-manuals */
    /* hvd   describing their codes. */
    /* hvd */
    /* hvd   This code was distributed through CCL (http://www.ccl.net/). */
    /* hvd */
    /* hvd   [1] V.I. Lebedev, and D.N. Laikov */
    /* hvd       "A quadrature formula for the sphere of the 131st */
    /* hvd        algebraic order of accuracy" */
    /* hvd       Doklady Mathematics, Vol. 59, No. 3, 1999, pp. 477-481. */
    /* hvd */
    /* hvd   [2] V.I. Lebedev */
    /* hvd       "A quadrature formula for the sphere of 59th algebraic */
    /* hvd        order of accuracy" */
    /* hvd       Russian Acad. Sci. Dokl. Math., Vol. 50, 1995, pp. 283-286. */
    /* hvd */
    /* hvd   [3] V.I. Lebedev, and A.L. Skorokhodov */
    /* hvd       "Quadrature formulas of orders 41, 47, and 53 for the sphere" */
    /* hvd       Russian Acad. Sci. Dokl. Math., Vol. 45, 1992, pp. 587-592. */
    /* hvd */
    /* hvd   [4] V.I. Lebedev */
    /* hvd       "Spherical quadrature formulas exact to orders 25-29" */
    /* hvd       Siberian Mathematical Journal, Vol. 18, 1977, pp. 99-107. */
    /* hvd */
    /* hvd   [5] V.I. Lebedev */
    /* hvd       "Quadratures on a sphere" */
    /* hvd       Computational Mathematics and Mathematical Physics, Vol. 16, */
    /* hvd       1976, pp. 10-24. */
    /* hvd */
    /* hvd   [6] V.I. Lebedev */
    /* hvd       "Values of the nodes and weights of ninth to seventeenth */
    /* hvd        order Gauss-Markov quadrature formulae invariant under the */
    /* hvd        octahedron group with inversion" */
    /* hvd       Computational Mathematics and Mathematical Physics, Vol. 15, */
    /* hvd       1975, pp. 44-51. */
    /* hvd */
    /* Parameter adjustments */
    --w;
    --z;
    --y;
    --x;

    /* Function Body */
    *n = 1;
    v = 2.192942088181184e-4;
    gen_oh_(1, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    v = .00143643361731908;
    gen_oh_(2, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    v = .001421940344335877;
    gen_oh_(3, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .0508720441050236;
    v = 6.798123511050502e-4;
    gen_oh_(4, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .1228198790178831;
    v = 9.913184235294912e-4;
    gen_oh_(4, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .2026890814408786;
    v = .001180207833238949;
    gen_oh_(4, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .2847745156464294;
    v = .001296599602080921;
    gen_oh_(4, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .3656719078978026;
    v = .001365871427428316;
    gen_oh_(4, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .4428264886713469;
    v = .001402988604775325;
    gen_oh_(4, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .5140619627249735;
    v = .001418645563595609;
    gen_oh_(4, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .6306401219166803;
    v = .001421376741851662;
    gen_oh_(4, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .6716883332022612;
    v = .001423996475490962;
    gen_oh_(4, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .6979792685336881;
    v = .001431554042178567;
    gen_oh_(4, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .1446865674195309;
    v = 9.254401499865368e-4;
    gen_oh_(5, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .3390263475411216;
    v = .001250239995053509;
    gen_oh_(5, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .5335804651263506;
    v = .00139436584332923;
    gen_oh_(5, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .06944024393349413;
    b = .2355187894242326;
    v = .001127089094671749;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .226900410952946;
    b = .410218247404573;
    v = .00134575376091067;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .08025574607775339;
    b = .6214302417481605;
    v = .001424957283316783;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .1467999527896572;
    b = .3245284345717394;
    v = .00126152334123775;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .1571507769824727;
    b = .522448218969663;
    v = .001392547106052696;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .2365702993157246;
    b = .6017546634089558;
    v = .001418761677877656;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .07714815866765732;
    b = .4346575516141163;
    v = .001338366684479554;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .306293666621073;
    b = .4908826589037616;
    v = .001393700862676131;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .3822477379524787;
    b = .56487681490995;
    v = .001415914757466932;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    --(*n);
    return 0;
}

int ld0974_(double *x, double *y, double *z, double *w, int *n)
{
    static double a, b, v;

    /* VW */
    /* VW    LEBEDEV  974-POINT ANGULAR GRID */
    /* VW */
    /* hvd */
    /* hvd   This subroutine is part of a set of subroutines that generate */
    /* hvd   Lebedev grids [1-6] for integration on a sphere. The original */
    /* hvd   C-code [1] was kindly provided by Dr. Dmitri N. Laikov and */
    /* hvd   translated into fortran by Dr. Christoph van Wuellen. */
    /* hvd   This subroutine was translated using a C to fortran77 conversion */
    /* hvd   tool written by Dr. Christoph van Wuellen. */
    /* hvd */
    /* hvd   Users of this code are asked to include reference [1] in their */
    /* hvd   publications, and in the user- and programmers-manuals */
    /* hvd   describing their codes. */
    /* hvd */
    /* hvd   This code was distributed through CCL (http://www.ccl.net/). */
    /* hvd */
    /* hvd   [1] V.I. Lebedev, and D.N. Laikov */
    /* hvd       "A quadrature formula for the sphere of the 131st */
    /* hvd        algebraic order of accuracy" */
    /* hvd       Doklady Mathematics, Vol. 59, No. 3, 1999, pp. 477-481. */
    /* hvd */
    /* hvd   [2] V.I. Lebedev */
    /* hvd       "A quadrature formula for the sphere of 59th algebraic */
    /* hvd        order of accuracy" */
    /* hvd       Russian Acad. Sci. Dokl. Math., Vol. 50, 1995, pp. 283-286. */
    /* hvd */
    /* hvd   [3] V.I. Lebedev, and A.L. Skorokhodov */
    /* hvd       "Quadrature formulas of orders 41, 47, and 53 for the sphere" */
    /* hvd       Russian Acad. Sci. Dokl. Math., Vol. 45, 1992, pp. 587-592. */
    /* hvd */
    /* hvd   [4] V.I. Lebedev */
    /* hvd       "Spherical quadrature formulas exact to orders 25-29" */
    /* hvd       Siberian Mathematical Journal, Vol. 18, 1977, pp. 99-107. */
    /* hvd */
    /* hvd   [5] V.I. Lebedev */
    /* hvd       "Quadratures on a sphere" */
    /* hvd       Computational Mathematics and Mathematical Physics, Vol. 16, */
    /* hvd       1976, pp. 10-24. */
    /* hvd */
    /* hvd   [6] V.I. Lebedev */
    /* hvd       "Values of the nodes and weights of ninth to seventeenth */
    /* hvd        order Gauss-Markov quadrature formulae invariant under the */
    /* hvd        octahedron group with inversion" */
    /* hvd       Computational Mathematics and Mathematical Physics, Vol. 15, */
    /* hvd       1975, pp. 44-51. */
    /* hvd */
    /* Parameter adjustments */
    --w;
    --z;
    --y;
    --x;

    /* Function Body */
    *n = 1;
    v = 1.438294190527431e-4;
    gen_oh_(1, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    v = .001125772288287004;
    gen_oh_(3, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .04292963545341347;
    v = 4.948029341949241e-4;
    gen_oh_(4, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .1051426854086404;
    v = 7.35799010912547e-4;
    gen_oh_(4, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .1750024867623087;
    v = 8.889132771304384e-4;
    gen_oh_(4, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .2477653379650257;
    v = 9.888347838921435e-4;
    gen_oh_(4, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .3206567123955957;
    v = .001053299681709471;
    gen_oh_(4, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .3916520749849983;
    v = .001092778807014578;
    gen_oh_(4, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .4590825874187624;
    v = .001114389394063227;
    gen_oh_(4, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .5214563888415861;
    v = .001123724788051555;
    gen_oh_(4, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .6253170244654199;
    v = .001125239325243814;
    gen_oh_(4, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .663792674452317;
    v = .001126153271815905;
    gen_oh_(4, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .6910410398498301;
    v = .001130286931123841;
    gen_oh_(4, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .705290700745776;
    v = .001134986534363955;
    gen_oh_(4, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .123668676265799;
    v = 6.823367927109931e-4;
    gen_oh_(5, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .2940777114468387;
    v = 9.454158160447096e-4;
    gen_oh_(5, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .4697753849207649;
    v = .001074429975385679;
    gen_oh_(5, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .6334563241139567;
    v = .001129300086569132;
    gen_oh_(5, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .05974048614181342;
    b = .2029128752777523;
    v = 8.436884500901954e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .1375760408473636;
    b = .4602621942484054;
    v = .001075255720448885;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .3391016526336286;
    b = .5030673999662036;
    v = .001108577236864462;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .127167519143982;
    b = .2817606422442134;
    v = 9.566475323783357e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .2693120740413512;
    b = .4331561291720157;
    v = .001080663250717391;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .1419786452601918;
    b = .6256167358580814;
    v = .001126797131196295;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .06709284600738255;
    b = .3798395216859157;
    v = .001022568715358061;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .07057738183256172;
    b = .551750542142352;
    v = .001108960267713108;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .2783888477882155;
    b = .6029619156159187;
    v = .001122790653435766;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .1979578938917407;
    b = .3589606329589096;
    v = .00103240184711746;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .2087307061103274;
    b = .5348666438135476;
    v = .001107249382283854;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .4055122137872836;
    b = .5674997546074373;
    v = .001121780048519972;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    --(*n);
    return 0;
}

int ld1202_(double *x, double *y, double *z, double *w, int *n)
{
    static double a, b, v;

    /* VW */
    /* VW    LEBEDEV 1202-POINT ANGULAR GRID */
    /* VW */
    /* hvd */
    /* hvd   This subroutine is part of a set of subroutines that generate */
    /* hvd   Lebedev grids [1-6] for integration on a sphere. The original */
    /* hvd   C-code [1] was kindly provided by Dr. Dmitri N. Laikov and */
    /* hvd   translated into fortran by Dr. Christoph van Wuellen. */
    /* hvd   This subroutine was translated using a C to fortran77 conversion */
    /* hvd   tool written by Dr. Christoph van Wuellen. */
    /* hvd */
    /* hvd   Users of this code are asked to include reference [1] in their */
    /* hvd   publications, and in the user- and programmers-manuals */
    /* hvd   describing their codes. */
    /* hvd */
    /* hvd   This code was distributed through CCL (http://www.ccl.net/). */
    /* hvd */
    /* hvd   [1] V.I. Lebedev, and D.N. Laikov */
    /* hvd       "A quadrature formula for the sphere of the 131st */
    /* hvd        algebraic order of accuracy" */
    /* hvd       Doklady Mathematics, Vol. 59, No. 3, 1999, pp. 477-481. */
    /* hvd */
    /* hvd   [2] V.I. Lebedev */
    /* hvd       "A quadrature formula for the sphere of 59th algebraic */
    /* hvd        order of accuracy" */
    /* hvd       Russian Acad. Sci. Dokl. Math., Vol. 50, 1995, pp. 283-286. */
    /* hvd */
    /* hvd   [3] V.I. Lebedev, and A.L. Skorokhodov */
    /* hvd       "Quadrature formulas of orders 41, 47, and 53 for the sphere" */
    /* hvd       Russian Acad. Sci. Dokl. Math., Vol. 45, 1992, pp. 587-592. */
    /* hvd */
    /* hvd   [4] V.I. Lebedev */
    /* hvd       "Spherical quadrature formulas exact to orders 25-29" */
    /* hvd       Siberian Mathematical Journal, Vol. 18, 1977, pp. 99-107. */
    /* hvd */
    /* hvd   [5] V.I. Lebedev */
    /* hvd       "Quadratures on a sphere" */
    /* hvd       Computational Mathematics and Mathematical Physics, Vol. 16, */
    /* hvd       1976, pp. 10-24. */
    /* hvd */
    /* hvd   [6] V.I. Lebedev */
    /* hvd       "Values of the nodes and weights of ninth to seventeenth */
    /* hvd        order Gauss-Markov quadrature formulae invariant under the */
    /* hvd        octahedron group with inversion" */
    /* hvd       Computational Mathematics and Mathematical Physics, Vol. 15, */
    /* hvd       1975, pp. 44-51. */
    /* hvd */
    /* Parameter adjustments */
    --w;
    --z;
    --y;
    --x;

    /* Function Body */
    *n = 1;
    v = 1.105189233267572e-4;
    gen_oh_(1, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    v = 9.205232738090741e-4;
    gen_oh_(2, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    v = 9.133159786443561e-4;
    gen_oh_(3, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .03712636449657089;
    v = 3.690421898017899e-4;
    gen_oh_(4, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .09140060412262223;
    v = 5.60399092868066e-4;
    gen_oh_(4, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .1531077852469906;
    v = 6.865297629282609e-4;
    gen_oh_(4, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .2180928891660612;
    v = 7.72033855114563e-4;
    gen_oh_(4, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .2839874532200175;
    v = 8.301545958894795e-4;
    gen_oh_(4, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .3491177600963764;
    v = 8.686692550179628e-4;
    gen_oh_(4, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .4121431461444309;
    v = 8.92707628584689e-4;
    gen_oh_(4, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .4718993627149127;
    v = 9.060820238568219e-4;
    gen_oh_(4, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .5273145452842337;
    v = 9.119777254940867e-4;
    gen_oh_(4, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .6209475332444019;
    v = 9.128720138604181e-4;
    gen_oh_(4, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .6569722711857291;
    v = 9.130714935691735e-4;
    gen_oh_(4, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .6841788309070143;
    v = 9.152873784554116e-4;
    gen_oh_(4, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .7012604330123631;
    v = 9.187436274321654e-4;
    gen_oh_(4, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .1072382215478166;
    v = 5.176977312965694e-4;
    gen_oh_(5, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .2582068959496968;
    v = 7.331143682101417e-4;
    gen_oh_(5, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .4172752955306717;
    v = 8.463232836379928e-4;
    gen_oh_(5, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .5700366911792503;
    v = 9.031122694253992e-4;
    gen_oh_(5, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .9827986018263947;
    b = .1771774022615325;
    v = 6.485778453163257e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .9624249230326228;
    b = .2475716463426288;
    v = 7.435030910982369e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .9402007994128811;
    b = .3354616289066489;
    v = 7.998527891839054e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .9320822040143202;
    b = .3173615246611977;
    v = 8.101731497468018e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .9043674199393299;
    b = .4090268427085357;
    v = 8.483389574594331e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .8912407560074747;
    b = .3854291150669224;
    v = 8.556299257311812e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .8676435628462708;
    b = .4932221184851285;
    v = 8.80320867973826e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .8581979986041619;
    b = .4785320675922435;
    v = 8.81104818242572e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .8396753624049856;
    b = .4507422593157064;
    v = 8.850282341265444e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .8165288564022188;
    b = .56321230207621;
    v = 9.021342299040653e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .8015469370783529;
    b = .54343035696939;
    v = 9.010091677105086e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .7773563069070351;
    b = .5123518486419871;
    v = 9.022692938426915e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .7661621213900394;
    b = .6394279634749102;
    v = 9.158016174693465e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .755358414353351;
    b = .6269805509024392;
    v = 9.131578003189435e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .7344305757559503;
    b = .603116169309631;
    v = 9.107813579482705e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .7043837184021765;
    b = .5693702498468441;
    v = 9.105760258970126e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    --(*n);
    return 0;
}

int ld1454_(double *x, double *y, double *z, double *w, int *n)
{
    static double a, b, v;

    /* VW */
    /* VW    LEBEDEV 1454-POINT ANGULAR GRID */
    /* VW */
    /* hvd */
    /* hvd   This subroutine is part of a set of subroutines that generate */
    /* hvd   Lebedev grids [1-6] for integration on a sphere. The original */
    /* hvd   C-code [1] was kindly provided by Dr. Dmitri N. Laikov and */
    /* hvd   translated into fortran by Dr. Christoph van Wuellen. */
    /* hvd   This subroutine was translated using a C to fortran77 conversion */
    /* hvd   tool written by Dr. Christoph van Wuellen. */
    /* hvd */
    /* hvd   Users of this code are asked to include reference [1] in their */
    /* hvd   publications, and in the user- and programmers-manuals */
    /* hvd   describing their codes. */
    /* hvd */
    /* hvd   This code was distributed through CCL (http://www.ccl.net/). */
    /* hvd */
    /* hvd   [1] V.I. Lebedev, and D.N. Laikov */
    /* hvd       "A quadrature formula for the sphere of the 131st */
    /* hvd        algebraic order of accuracy" */
    /* hvd       Doklady Mathematics, Vol. 59, No. 3, 1999, pp. 477-481. */
    /* hvd */
    /* hvd   [2] V.I. Lebedev */
    /* hvd       "A quadrature formula for the sphere of 59th algebraic */
    /* hvd        order of accuracy" */
    /* hvd       Russian Acad. Sci. Dokl. Math., Vol. 50, 1995, pp. 283-286. */
    /* hvd */
    /* hvd   [3] V.I. Lebedev, and A.L. Skorokhodov */
    /* hvd       "Quadrature formulas of orders 41, 47, and 53 for the sphere" */
    /* hvd       Russian Acad. Sci. Dokl. Math., Vol. 45, 1992, pp. 587-592. */
    /* hvd */
    /* hvd   [4] V.I. Lebedev */
    /* hvd       "Spherical quadrature formulas exact to orders 25-29" */
    /* hvd       Siberian Mathematical Journal, Vol. 18, 1977, pp. 99-107. */
    /* hvd */
    /* hvd   [5] V.I. Lebedev */
    /* hvd       "Quadratures on a sphere" */
    /* hvd       Computational Mathematics and Mathematical Physics, Vol. 16, */
    /* hvd       1976, pp. 10-24. */
    /* hvd */
    /* hvd   [6] V.I. Lebedev */
    /* hvd       "Values of the nodes and weights of ninth to seventeenth */
    /* hvd        order Gauss-Markov quadrature formulae invariant under the */
    /* hvd        octahedron group with inversion" */
    /* hvd       Computational Mathematics and Mathematical Physics, Vol. 15, */
    /* hvd       1975, pp. 44-51. */
    /* hvd */
    /* Parameter adjustments */
    --w;
    --z;
    --y;
    --x;

    /* Function Body */
    *n = 1;
    v = 7.777160743261247e-5;
    gen_oh_(1, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    v = 7.557646413004701e-4;
    gen_oh_(3, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .03229290663413854;
    v = 2.841633806090617e-4;
    gen_oh_(4, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .08036733271462222;
    v = 4.374419127053555e-4;
    gen_oh_(4, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .1354289960531653;
    v = 5.417174740872172e-4;
    gen_oh_(4, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .1938963861114426;
    v = 6.148000891358593e-4;
    gen_oh_(4, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .2537343715011275;
    v = 6.664394485800705e-4;
    gen_oh_(4, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .313525143475257;
    v = 7.02503935692322e-4;
    gen_oh_(4, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .3721558339375338;
    v = 7.268511789249627e-4;
    gen_oh_(4, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .4286809575195696;
    v = 7.422637534208629e-4;
    gen_oh_(4, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .4822510128282994;
    v = 7.509545035841214e-4;
    gen_oh_(4, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .5320679333566263;
    v = 7.548535057718401e-4;
    gen_oh_(4, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .6172998195394274;
    v = 7.554088969774001e-4;
    gen_oh_(4, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .6510679849127481;
    v = 7.553147174442808e-4;
    gen_oh_(4, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .677731525168736;
    v = 7.564767653292297e-4;
    gen_oh_(4, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .6963109410648741;
    v = 7.58799180851873e-4;
    gen_oh_(4, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .7058935009831749;
    v = 7.608261832033027e-4;
    gen_oh_(4, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .9955546194091857;
    v = 4.021680447874916e-4;
    gen_oh_(5, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .9734115901794209;
    v = 5.804871793945964e-4;
    gen_oh_(5, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .9275693732388626;
    v = 6.792151955945159e-4;
    gen_oh_(5, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .8568022422795103;
    v = 7.336741211286294e-4;
    gen_oh_(5, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .7623495553719372;
    v = 7.581866300989608e-4;
    gen_oh_(5, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .5707522908892223;
    b = .4387028039889501;
    v = 7.538257859800743e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .5196463388403083;
    b = .3858908414762617;
    v = 7.483517247053123e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .4646337531215351;
    b = .3301937372343854;
    v = 7.371763661112059e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .4063901697557691;
    b = .2725423573563777;
    v = 7.183448895756934e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .3456329466643087;
    b = .213951023749525;
    v = 6.895815529822191e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .2831395121050332;
    b = .1555922309786647;
    v = 6.480105801792886e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .219768202292533;
    b = .09892878979686097;
    v = 5.897558896594636e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .1564696098650355;
    b = .0459864291067551;
    v = 5.095708849247346e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .6027356673721295;
    b = .3376625140173426;
    v = 7.536906428909755e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .5496032320255096;
    b = .2822301309727988;
    v = 7.472505965575118e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .4921707755234567;
    b = .224863234259254;
    v = 7.343017132279698e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .4309422998598483;
    b = .1666224723456479;
    v = 7.130871582177445e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .3664108182313672;
    b = .1086964901822169;
    v = 6.817022032112776e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .2990189057758436;
    b = .05251989784120085;
    v = 6.380941145604121e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .6268724013144998;
    b = .2297523657550023;
    v = 7.55038137792031e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .5707324144834607;
    b = .17230806070938;
    v = 7.478646640144802e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .5096360901960365;
    b = .1140238465390513;
    v = 7.33591872060122e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .4438729938312456;
    b = .05611522095882537;
    v = 7.110120527658118e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .6419978471082389;
    b = .1164174423140873;
    v = 7.571363978689501e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .5817218061802611;
    b = .05797589531445219;
    v = 7.489908329079234e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    --(*n);
    return 0;
}

int ld1730_(double *x, double *y, double *z, double *w, int *n)
{
    static double a, b, v;

    /* VW */
    /* VW    LEBEDEV 1730-POINT ANGULAR GRID */
    /* VW */
    /* hvd */
    /* hvd   This subroutine is part of a set of subroutines that generate */
    /* hvd   Lebedev grids [1-6] for integration on a sphere. The original */
    /* hvd   C-code [1] was kindly provided by Dr. Dmitri N. Laikov and */
    /* hvd   translated into fortran by Dr. Christoph van Wuellen. */
    /* hvd   This subroutine was translated using a C to fortran77 conversion */
    /* hvd   tool written by Dr. Christoph van Wuellen. */
    /* hvd */
    /* hvd   Users of this code are asked to include reference [1] in their */
    /* hvd   publications, and in the user- and programmers-manuals */
    /* hvd   describing their codes. */
    /* hvd */
    /* hvd   This code was distributed through CCL (http://www.ccl.net/). */
    /* hvd */
    /* hvd   [1] V.I. Lebedev, and D.N. Laikov */
    /* hvd       "A quadrature formula for the sphere of the 131st */
    /* hvd        algebraic order of accuracy" */
    /* hvd       Doklady Mathematics, Vol. 59, No. 3, 1999, pp. 477-481. */
    /* hvd */
    /* hvd   [2] V.I. Lebedev */
    /* hvd       "A quadrature formula for the sphere of 59th algebraic */
    /* hvd        order of accuracy" */
    /* hvd       Russian Acad. Sci. Dokl. Math., Vol. 50, 1995, pp. 283-286. */
    /* hvd */
    /* hvd   [3] V.I. Lebedev, and A.L. Skorokhodov */
    /* hvd       "Quadrature formulas of orders 41, 47, and 53 for the sphere" */
    /* hvd       Russian Acad. Sci. Dokl. Math., Vol. 45, 1992, pp. 587-592. */
    /* hvd */
    /* hvd   [4] V.I. Lebedev */
    /* hvd       "Spherical quadrature formulas exact to orders 25-29" */
    /* hvd       Siberian Mathematical Journal, Vol. 18, 1977, pp. 99-107. */
    /* hvd */
    /* hvd   [5] V.I. Lebedev */
    /* hvd       "Quadratures on a sphere" */
    /* hvd       Computational Mathematics and Mathematical Physics, Vol. 16, */
    /* hvd       1976, pp. 10-24. */
    /* hvd */
    /* hvd   [6] V.I. Lebedev */
    /* hvd       "Values of the nodes and weights of ninth to seventeenth */
    /* hvd        order Gauss-Markov quadrature formulae invariant under the */
    /* hvd        octahedron group with inversion" */
    /* hvd       Computational Mathematics and Mathematical Physics, Vol. 15, */
    /* hvd       1975, pp. 44-51. */
    /* hvd */
    /* Parameter adjustments */
    --w;
    --z;
    --y;
    --x;

    /* Function Body */
    *n = 1;
    v = 6.309049437420976e-5;
    gen_oh_(1, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    v = 6.398287705571748e-4;
    gen_oh_(2, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    v = 6.35718507353072e-4;
    gen_oh_(3, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .02860923126194662;
    v = 2.221207162188168e-4;
    gen_oh_(4, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .07142556767711522;
    v = 3.475784022286848e-4;
    gen_oh_(4, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .1209199540995559;
    v = 4.350742443589804e-4;
    gen_oh_(4, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .1738673106594379;
    v = 4.978569136522127e-4;
    gen_oh_(4, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .2284645438467734;
    v = 5.435036221998053e-4;
    gen_oh_(4, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .2834807671701512;
    v = 5.765913388219542e-4;
    gen_oh_(4, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .3379680145467339;
    v = 6.001200359226003e-4;
    gen_oh_(4, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .3911355454819537;
    v = 6.162178172717512e-4;
    gen_oh_(4, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .4422860353001403;
    v = 6.265218152438485e-4;
    gen_oh_(4, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .4907781568726057;
    v = 6.323987160974212e-4;
    gen_oh_(4, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .5360006153211468;
    v = 6.350767851540569e-4;
    gen_oh_(4, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .6142105973596603;
    v = 6.354362775297107e-4;
    gen_oh_(4, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .6459300387977504;
    v = 6.352302462706235e-4;
    gen_oh_(4, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .6718056125089225;
    v = 6.358117881417972e-4;
    gen_oh_(4, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .6910888533186254;
    v = 6.373101590310117e-4;
    gen_oh_(4, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .7030467416823252;
    v = 6.390428961368665e-4;
    gen_oh_(4, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .08354951166354646;
    v = 3.186913449946576e-4;
    gen_oh_(5, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .2050143009099486;
    v = 4.678028558591711e-4;
    gen_oh_(5, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .3370208290706637;
    v = 5.538829697598626e-4;
    gen_oh_(5, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .4689051484233963;
    v = 6.044475907190476e-4;
    gen_oh_(5, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .5939400424557334;
    v = 6.313575103509012e-4;
    gen_oh_(5, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .1394983311832261;
    b = .04097581162050343;
    v = 4.07862643185563e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .1967999180485014;
    b = .08851987391293348;
    v = 4.759933057812725e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .2546183732548967;
    b = .1397680182969819;
    v = 5.26815118641344e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .3121281074713875;
    b = .1929452542226526;
    v = 5.643048560507316e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .3685981078502492;
    b = .2467898337061562;
    v = 5.914501076613073e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .4233760321547856;
    b = .3003104124785409;
    v = 6.104561257874195e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .4758671236059246;
    b = .3526684328175033;
    v = 6.230252860707806e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .5255178579796463;
    b = .4031134861145713;
    v = 6.305618761760796e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .5718025633734589;
    b = .4509426448342351;
    v = 6.343092767597889e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .2686927772723415;
    b = .04711322502423248;
    v = 5.176268945737826e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .3306006819904809;
    b = .09784487303942695;
    v = 5.564840313313692e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .3904906850594983;
    b = .1505395810025273;
    v = 5.85642667103898e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .447995795190439;
    b = .203972815629605;
    v = 6.066386925777091e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .502707684891978;
    b = .2571529941121107;
    v = 6.208824962234458e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .5542087392260217;
    b = .309219137581567;
    v = 6.296314297822907e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .6020850887375187;
    b = .3593807506130276;
    v = 6.340423756791859e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .4019851409179594;
    b = .05063389934378671;
    v = 5.829627677107342e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .46356145674498;
    b = .1032422269160612;
    v = 6.04869337608111e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .5215860931591575;
    b = .1566322094006254;
    v = 6.202362317732461e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .5758202499099271;
    b = .2098082827491099;
    v = 6.299005328403779e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .6259893683876795;
    b = .2618824114553391;
    v = 6.347722390609353e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .5313795124811891;
    b = .05263245019338556;
    v = 6.203778981238834e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .5893317955931995;
    b = .1061059730982005;
    v = 6.308414671239979e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .6426246321215801;
    b = .1594171564034221;
    v = 6.362706466959498e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .6511904367376113;
    b = .0535478953656554;
    v = 6.375414170333233e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    --(*n);
    return 0;
}

int ld2030_(double *x, double *y, double *z, double *w, int *n)
{
    static double a, b, v;

    /* VW */
    /* VW    LEBEDEV 2030-POINT ANGULAR GRID */
    /* VW */
    /* hvd */
    /* hvd   This subroutine is part of a set of subroutines that generate */
    /* hvd   Lebedev grids [1-6] for integration on a sphere. The original */
    /* hvd   C-code [1] was kindly provided by Dr. Dmitri N. Laikov and */
    /* hvd   translated into fortran by Dr. Christoph van Wuellen. */
    /* hvd   This subroutine was translated using a C to fortran77 conversion */
    /* hvd   tool written by Dr. Christoph van Wuellen. */
    /* hvd */
    /* hvd   Users of this code are asked to include reference [1] in their */
    /* hvd   publications, and in the user- and programmers-manuals */
    /* hvd   describing their codes. */
    /* hvd */
    /* hvd   This code was distributed through CCL (http://www.ccl.net/). */
    /* hvd */
    /* hvd   [1] V.I. Lebedev, and D.N. Laikov */
    /* hvd       "A quadrature formula for the sphere of the 131st */
    /* hvd        algebraic order of accuracy" */
    /* hvd       Doklady Mathematics, Vol. 59, No. 3, 1999, pp. 477-481. */
    /* hvd */
    /* hvd   [2] V.I. Lebedev */
    /* hvd       "A quadrature formula for the sphere of 59th algebraic */
    /* hvd        order of accuracy" */
    /* hvd       Russian Acad. Sci. Dokl. Math., Vol. 50, 1995, pp. 283-286. */
    /* hvd */
    /* hvd   [3] V.I. Lebedev, and A.L. Skorokhodov */
    /* hvd       "Quadrature formulas of orders 41, 47, and 53 for the sphere" */
    /* hvd       Russian Acad. Sci. Dokl. Math., Vol. 45, 1992, pp. 587-592. */
    /* hvd */
    /* hvd   [4] V.I. Lebedev */
    /* hvd       "Spherical quadrature formulas exact to orders 25-29" */
    /* hvd       Siberian Mathematical Journal, Vol. 18, 1977, pp. 99-107. */
    /* hvd */
    /* hvd   [5] V.I. Lebedev */
    /* hvd       "Quadratures on a sphere" */
    /* hvd       Computational Mathematics and Mathematical Physics, Vol. 16, */
    /* hvd       1976, pp. 10-24. */
    /* hvd */
    /* hvd   [6] V.I. Lebedev */
    /* hvd       "Values of the nodes and weights of ninth to seventeenth */
    /* hvd        order Gauss-Markov quadrature formulae invariant under the */
    /* hvd        octahedron group with inversion" */
    /* hvd       Computational Mathematics and Mathematical Physics, Vol. 15, */
    /* hvd       1975, pp. 44-51. */
    /* hvd */
    /* Parameter adjustments */
    --w;
    --z;
    --y;
    --x;

    /* Function Body */
    *n = 1;
    v = 4.656031899197431e-5;
    gen_oh_(1, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    v = 5.421549195295507e-4;
    gen_oh_(3, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .02540835336814348;
    v = 1.778522133346553e-4;
    gen_oh_(4, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .06399322800504915;
    v = 2.811325405682796e-4;
    gen_oh_(4, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .1088269469804125;
    v = 3.548896312631459e-4;
    gen_oh_(4, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .1570670798818287;
    v = 4.090310897173364e-4;
    gen_oh_(4, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .2071163932282514;
    v = 4.493286134169965e-4;
    gen_oh_(4, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .2578914044450844;
    v = 4.793728447962723e-4;
    gen_oh_(4, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .3085687558169623;
    v = 5.015415319164265e-4;
    gen_oh_(4, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .3584719706267024;
    v = 5.175127372677937e-4;
    gen_oh_(4, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .4070135594428709;
    v = 5.285522262081019e-4;
    gen_oh_(4, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .4536618626222638;
    v = 5.356832703713962e-4;
    gen_oh_(4, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .4979195686463577;
    v = 5.39791473617517e-4;
    gen_oh_(4, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .5393075111126999;
    v = 5.41689944159993e-4;
    gen_oh_(4, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .6115617676843916;
    v = 5.419308476889938e-4;
    gen_oh_(4, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .6414308435160159;
    v = 5.416936902030596e-4;
    gen_oh_(4, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .6664099412721607;
    v = 5.419544338703164e-4;
    gen_oh_(4, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .6859161771214913;
    v = 5.428983656630975e-4;
    gen_oh_(4, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .699362559350389;
    v = 5.442286500098193e-4;
    gen_oh_(4, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .706239338771938;
    v = 5.452250345057301e-4;
    gen_oh_(4, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .07479028168349763;
    v = 2.56800249772853e-4;
    gen_oh_(5, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .1848951153969366;
    v = 3.827211700292145e-4;
    gen_oh_(5, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .3059529066581305;
    v = 4.579491561917824e-4;
    gen_oh_(5, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .4285556101021362;
    v = 5.042003969083574e-4;
    gen_oh_(5, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .5468758653496526;
    v = 5.312708889976025e-4;
    gen_oh_(5, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .6565821978343439;
    v = 5.438401790747117e-4;
    gen_oh_(5, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .1253901572367117;
    b = .03681917226439641;
    v = 3.316041873197344e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .1775721510383941;
    b = .07982487607213301;
    v = 3.899113567153771e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .2305693358216114;
    b = .1264640966592335;
    v = 4.343343327201309e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .2836502845992063;
    b = .1751585683418957;
    v = 4.679415262318919e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .336179474623259;
    b = .224799590763267;
    v = 4.930847981631031e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .3875979172264824;
    b = .2745299257422246;
    v = 5.115031867540091e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .4374019316999074;
    b = .3236373482441118;
    v = 5.245217148457367e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .4851275843340022;
    b = .3714967859436741;
    v = 5.332041499895321e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .5303391803806868;
    b = .4175353646321745;
    v = 5.384583126021542e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .5726197380596287;
    b = .4612084406355461;
    v = 5.411067210798852e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .2431520732564863;
    b = .04258040133043952;
    v = 4.259797391468714e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .3002096800895869;
    b = .08869424306722721;
    v = 4.604931368460021e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .3558554457457432;
    b = .1368811706510655;
    v = 4.871814878255202e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .4097782537048887;
    b = .1860739985015033;
    v = 5.072242910074885e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .4616337666067458;
    b = .2354235077395853;
    v = 5.21706984523535e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .5110707008417874;
    b = .2842074921347011;
    v = 5.31578596628031e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .5577415286163795;
    b = .3317784414984102;
    v = 5.376833708758905e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .601306043136695;
    b = .37752990020407;
    v = 5.408032092069521e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .3661596767261781;
    b = .04599367887164592;
    v = 4.842744917904866e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .4237633153506581;
    b = .09404893773654421;
    v = 5.04892607618813e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .4786328454658452;
    b = .1431377109091971;
    v = 5.202607980478373e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .5305702076789774;
    b = .192418638884357;
    v = 5.309932388325743e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .5793436224231788;
    b = .241159094477519;
    v = 5.377419770895208e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .6247069017094747;
    b = .2886871491583605;
    v = 5.411696331677717e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .4874315552535204;
    b = .04804978774953206;
    v = 5.19799629328242e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .5427337322059053;
    b = .09716857199366665;
    v = 5.311120836622945e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .59434937472467;
    b = .1465205839795055;
    v = 5.384309319956951e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .6421314033564943;
    b = .1953579449803574;
    v = 5.421859504051886e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .602062837471398;
    b = .04916375015738108;
    v = 5.390948355046314e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .6529222529856881;
    b = .09861621540127005;
    v = 5.433312705027845e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    --(*n);
    return 0;
}

int ld2354_(double *x, double *y, double *z, double *w, int *n)
{
    static double a, b, v;

    /* VW */
    /* VW    LEBEDEV 2354-POINT ANGULAR GRID */
    /* VW */
    /* hvd */
    /* hvd   This subroutine is part of a set of subroutines that generate */
    /* hvd   Lebedev grids [1-6] for integration on a sphere. The original */
    /* hvd   C-code [1] was kindly provided by Dr. Dmitri N. Laikov and */
    /* hvd   translated into fortran by Dr. Christoph van Wuellen. */
    /* hvd   This subroutine was translated using a C to fortran77 conversion */
    /* hvd   tool written by Dr. Christoph van Wuellen. */
    /* hvd */
    /* hvd   Users of this code are asked to include reference [1] in their */
    /* hvd   publications, and in the user- and programmers-manuals */
    /* hvd   describing their codes. */
    /* hvd */
    /* hvd   This code was distributed through CCL (http://www.ccl.net/). */
    /* hvd */
    /* hvd   [1] V.I. Lebedev, and D.N. Laikov */
    /* hvd       "A quadrature formula for the sphere of the 131st */
    /* hvd        algebraic order of accuracy" */
    /* hvd       Doklady Mathematics, Vol. 59, No. 3, 1999, pp. 477-481. */
    /* hvd */
    /* hvd   [2] V.I. Lebedev */
    /* hvd       "A quadrature formula for the sphere of 59th algebraic */
    /* hvd        order of accuracy" */
    /* hvd       Russian Acad. Sci. Dokl. Math., Vol. 50, 1995, pp. 283-286. */
    /* hvd */
    /* hvd   [3] V.I. Lebedev, and A.L. Skorokhodov */
    /* hvd       "Quadrature formulas of orders 41, 47, and 53 for the sphere" */
    /* hvd       Russian Acad. Sci. Dokl. Math., Vol. 45, 1992, pp. 587-592. */
    /* hvd */
    /* hvd   [4] V.I. Lebedev */
    /* hvd       "Spherical quadrature formulas exact to orders 25-29" */
    /* hvd       Siberian Mathematical Journal, Vol. 18, 1977, pp. 99-107. */
    /* hvd */
    /* hvd   [5] V.I. Lebedev */
    /* hvd       "Quadratures on a sphere" */
    /* hvd       Computational Mathematics and Mathematical Physics, Vol. 16, */
    /* hvd       1976, pp. 10-24. */
    /* hvd */
    /* hvd   [6] V.I. Lebedev */
    /* hvd       "Values of the nodes and weights of ninth to seventeenth */
    /* hvd        order Gauss-Markov quadrature formulae invariant under the */
    /* hvd        octahedron group with inversion" */
    /* hvd       Computational Mathematics and Mathematical Physics, Vol. 15, */
    /* hvd       1975, pp. 44-51. */
    /* hvd */
    /* Parameter adjustments */
    --w;
    --z;
    --y;
    --x;

    /* Function Body */
    *n = 1;
    v = 3.922616270665292e-5;
    gen_oh_(1, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    v = 4.703831750854424e-4;
    gen_oh_(2, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    v = 4.678202801282136e-4;
    gen_oh_(3, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .02290024646530589;
    v = 1.4378322289799e-4;
    gen_oh_(4, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .05779086652271284;
    v = 2.303572493577644e-4;
    gen_oh_(4, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .09863103576375984;
    v = 2.933110752447454e-4;
    gen_oh_(4, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .1428155792982185;
    v = 3.402905998359838e-4;
    gen_oh_(4, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .1888978116601463;
    v = 3.759138466870372e-4;
    gen_oh_(4, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .235909168297021;
    v = 4.030638447899798e-4;
    gen_oh_(4, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .2831228833706171;
    v = 4.236591432242211e-4;
    gen_oh_(4, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .3299495857966693;
    v = 4.390522656946746e-4;
    gen_oh_(4, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .3758840802660796;
    v = 4.502523466626247e-4;
    gen_oh_(4, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .420475183100948;
    v = 4.580577727783541e-4;
    gen_oh_(4, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .4633068518751051;
    v = 4.631391616615899e-4;
    gen_oh_(4, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .5039849474507313;
    v = 4.660928953698676e-4;
    gen_oh_(4, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .5421265793440747;
    v = 4.674751807936953e-4;
    gen_oh_(4, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .609266023055731;
    v = 4.67641490393292e-4;
    gen_oh_(4, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .6374654204984869;
    v = 4.67408649234787e-4;
    gen_oh_(4, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .6615136472609892;
    v = 4.674928539483207e-4;
    gen_oh_(4, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .6809487285958127;
    v = 4.680748979686447e-4;
    gen_oh_(4, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .6952980021665196;
    v = 4.69044980638904e-4;
    gen_oh_(4, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .70412454976954;
    v = 4.699877075860818e-4;
    gen_oh_(4, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .06744033088306065;
    v = 2.099942281069176e-4;
    gen_oh_(5, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .1678684485334166;
    v = 3.172269150712804e-4;
    gen_oh_(5, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .2793559049539613;
    v = 3.832051358546523e-4;
    gen_oh_(5, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .3935264218057639;
    v = 4.252193818146985e-4;
    gen_oh_(5, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .5052629268232558;
    v = 4.513807963755e-4;
    gen_oh_(5, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .6107905315437531;
    v = 4.657797469114178e-4;
    gen_oh_(5, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .1135081039843524;
    b = .03331954884662588;
    v = 2.733362800522836e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .1612866626099378;
    b = .07247167465436538;
    v = 3.235485368463559e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .2100786550168205;
    b = .1151539110849745;
    v = 3.624908726013453e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .2592282009459942;
    b = .1599491097143677;
    v = 3.925540070712828e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .3081740561320203;
    b = .2058699956028027;
    v = 4.156129781116235e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .3564289781578164;
    b = .2521624953502911;
    v = 4.330644984623263e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .4035587288240703;
    b = .2982090785797674;
    v = 4.459677725921312e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .4491671196373903;
    b = .3434762087235733;
    v = 4.551593004456795e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .4928854782917489;
    b = .3874831357203437;
    v = 4.613341462749918e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .5343646791958988;
    b = .4297814821746926;
    v = 4.651019618269806e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .573268321653099;
    b = .4699402260943537;
    v = 4.670249536100625e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .2214131583218986;
    b = .03873602040643895;
    v = 3.549555576441708e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .2741796504750071;
    b = .08089496256902013;
    v = 3.85610824524901e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .3259797439149485;
    b = .1251732177620872;
    v = 4.098622845756882e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .3765441148826891;
    b = .1706260286403185;
    v = 4.28632860426895e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .4255773574530558;
    b = .2165115147300408;
    v = 4.427802198993945e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .472779511705843;
    b = .2622089812225259;
    v = 4.530473511488561e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .5178546895819012;
    b = .3071721431296201;
    v = 4.600805475703138e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .560514119209746;
    b = .3508998998801138;
    v = 4.644599059958017e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .6004763319352512;
    b = .3929160876166931;
    v = 4.667274455712508e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .3352842634946949;
    b = .04202563457288019;
    v = 4.069360518020356e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .389197162981467;
    b = .0861430975887085;
    v = 4.260442819919195e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .4409875565542281;
    b = .1314500879380001;
    v = 4.408678508029063e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .4904893058592484;
    b = .1772189657383859;
    v = 4.518748115548597e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .5375056138769549;
    b = .2228277110050294;
    v = 4.595564875375116e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .5818255708669969;
    b = .2677179935014386;
    v = 4.643988774315846e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .6232334858144959;
    b = .3113675035544165;
    v = 4.668827491646946e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .4489485354492058;
    b = .04409162378368174;
    v = 4.400541823741973e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .501513687593315;
    b = .08939009917748489;
    v = 4.514512890193797e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .5511300550512623;
    b = .1351806029383365;
    v = 4.596198627347549e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .5976720409858;
    b = .1808370355053196;
    v = 4.648659016801781e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .6409956378989354;
    b = .2257852192301602;
    v = 4.675502017157673e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .5581222330827514;
    b = .0453217342163716;
    v = 4.598494476455523e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .6074705984161695;
    b = .09117488031840314;
    v = 4.654916955152048e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .6532272537379033;
    b = .1369294213140155;
    v = 4.684709779505137e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .6594761494500487;
    b = .04589901487275583;
    v = 4.691445539106986e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    --(*n);
    return 0;
}

int ld2702_(double *x, double *y, double *z, double *w, int *n)
{
    static double a, b, v;

    /* VW */
    /* VW    LEBEDEV 2702-POINT ANGULAR GRID */
    /* VW */
    /* hvd */
    /* hvd   This subroutine is part of a set of subroutines that generate */
    /* hvd   Lebedev grids [1-6] for integration on a sphere. The original */
    /* hvd   C-code [1] was kindly provided by Dr. Dmitri N. Laikov and */
    /* hvd   translated into fortran by Dr. Christoph van Wuellen. */
    /* hvd   This subroutine was translated using a C to fortran77 conversion */
    /* hvd   tool written by Dr. Christoph van Wuellen. */
    /* hvd */
    /* hvd   Users of this code are asked to include reference [1] in their */
    /* hvd   publications, and in the user- and programmers-manuals */
    /* hvd   describing their codes. */
    /* hvd */
    /* hvd   This code was distributed through CCL (http://www.ccl.net/). */
    /* hvd */
    /* hvd   [1] V.I. Lebedev, and D.N. Laikov */
    /* hvd       "A quadrature formula for the sphere of the 131st */
    /* hvd        algebraic order of accuracy" */
    /* hvd       Doklady Mathematics, Vol. 59, No. 3, 1999, pp. 477-481. */
    /* hvd */
    /* hvd   [2] V.I. Lebedev */
    /* hvd       "A quadrature formula for the sphere of 59th algebraic */
    /* hvd        order of accuracy" */
    /* hvd       Russian Acad. Sci. Dokl. Math., Vol. 50, 1995, pp. 283-286. */
    /* hvd */
    /* hvd   [3] V.I. Lebedev, and A.L. Skorokhodov */
    /* hvd       "Quadrature formulas of orders 41, 47, and 53 for the sphere" */
    /* hvd       Russian Acad. Sci. Dokl. Math., Vol. 45, 1992, pp. 587-592. */
    /* hvd */
    /* hvd   [4] V.I. Lebedev */
    /* hvd       "Spherical quadrature formulas exact to orders 25-29" */
    /* hvd       Siberian Mathematical Journal, Vol. 18, 1977, pp. 99-107. */
    /* hvd */
    /* hvd   [5] V.I. Lebedev */
    /* hvd       "Quadratures on a sphere" */
    /* hvd       Computational Mathematics and Mathematical Physics, Vol. 16, */
    /* hvd       1976, pp. 10-24. */
    /* hvd */
    /* hvd   [6] V.I. Lebedev */
    /* hvd       "Values of the nodes and weights of ninth to seventeenth */
    /* hvd        order Gauss-Markov quadrature formulae invariant under the */
    /* hvd        octahedron group with inversion" */
    /* hvd       Computational Mathematics and Mathematical Physics, Vol. 15, */
    /* hvd       1975, pp. 44-51. */
    /* hvd */
    /* Parameter adjustments */
    --w;
    --z;
    --y;
    --x;

    /* Function Body */
    *n = 1;
    v = 2.998675149888161e-5;
    gen_oh_(1, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    v = 4.077860529495355e-4;
    gen_oh_(3, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .02065562538818703;
    v = 1.185349192520667e-4;
    gen_oh_(4, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .05250918173022379;
    v = 1.913408643425751e-4;
    gen_oh_(4, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .08993480082038376;
    v = 2.452886577209897e-4;
    gen_oh_(4, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .1306023924436019;
    v = 2.862408183288702e-4;
    gen_oh_(4, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .1732060388531418;
    v = 3.178032258257357e-4;
    gen_oh_(4, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .2168727084820249;
    v = 3.42294566763369e-4;
    gen_oh_(4, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .2609528309173586;
    v = 3.612790520235922e-4;
    gen_oh_(4, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .3049252927938952;
    v = 3.758638229818521e-4;
    gen_oh_(4, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .3483484138084404;
    v = 3.868711798859953e-4;
    gen_oh_(4, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .3908321549106406;
    v = 3.949429933189938e-4;
    gen_oh_(4, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .4320210071894814;
    v = 4.006068107541156e-4;
    gen_oh_(4, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .4715824795890053;
    v = 4.043192149672723e-4;
    gen_oh_(4, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .5091984794078453;
    v = 4.064947495808078e-4;
    gen_oh_(4, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .5445580145650803;
    v = 4.075245619813152e-4;
    gen_oh_(4, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .6072575796841768;
    v = 4.076423540893566e-4;
    gen_oh_(4, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .6339484505755803;
    v = 4.074280862251555e-4;
    gen_oh_(4, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .6570718257486958;
    v = 4.074163756012244e-4;
    gen_oh_(4, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .6762557330090709;
    v = 4.077647795071246e-4;
    gen_oh_(4, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .691116169692379;
    v = 4.08451755278253e-4;
    gen_oh_(4, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .7012841911659961;
    v = 4.092468459224052e-4;
    gen_oh_(4, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .706455927241002;
    v = 4.097872687240906e-4;
    gen_oh_(4, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .06123554989894765;
    v = 1.738986811745028e-4;
    gen_oh_(5, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .1533070348312393;
    v = 2.659616045280191e-4;
    gen_oh_(5, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .2563902605244206;
    v = 3.240596008171533e-4;
    gen_oh_(5, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .3629346991663361;
    v = 3.621195964432943e-4;
    gen_oh_(5, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .4683949968987538;
    v = 3.868838330760539e-4;
    gen_oh_(5, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .5694479240657952;
    v = 4.018911532693111e-4;
    gen_oh_(5, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .6634465430993955;
    v = 4.089929432983252e-4;
    gen_oh_(5, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .1033958573552305;
    b = .03034544009063584;
    v = 2.279907527706409e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .1473521412414395;
    b = .06618803044247135;
    v = 2.715205490578897e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .1924552158705967;
    b = .1054431128987715;
    v = 3.057917896703976e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .2381094362890328;
    b = .1468263551238858;
    v = 3.326913052452555e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .283812170793676;
    b = .1894486108187886;
    v = 3.537334711890037e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .3291323133373415;
    b = .2326374238761579;
    v = 3.700567500783129e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .373689697874146;
    b = .2758485808485768;
    v = 3.825245372589122e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .4171406040760013;
    b = .3186179331996921;
    v = 3.918125171518296e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .4591677985256915;
    b = .3605329796303794;
    v = 3.984720419937579e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .4994733831718418;
    b = .4012147253586509;
    v = 4.029746003338211e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .5377731830445096;
    b = .4403050025570692;
    v = 4.057428632156627e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .5737917830001331;
    b = .4774565904277483;
    v = 4.071719274114857e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .2027323586271389;
    b = .03544122504976147;
    v = 2.990236950664119e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .2516942375187273;
    b = .07418304388646328;
    v = 3.262951734212878e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .3000227995257181;
    b = .1150502745727186;
    v = 3.482634608242413e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .3474806691046342;
    b = .1571963371209364;
    v = 3.656596681700892e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .3938103180359209;
    b = .19996318772471;
    v = 3.791740467794218e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .4387519590455703;
    b = .2428073457846535;
    v = 3.894034450156905e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .4820503960077787;
    b = .2852575132906155;
    v = 3.968600245508371e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .5234573778475101;
    b = .3268884208674639;
    v = 4.01993135142005e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .5627318647235282;
    b = .3673033321675939;
    v = 4.052108801278599e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .5996390607156954;
    b = .406121155183029;
    v = 4.068978613940934e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .3084780753791947;
    b = .03860125523100059;
    v = 3.454275351319704e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .3589988275920223;
    b = .07928938987104867;
    v = 3.62996353700792e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .4078628415881973;
    b = .1212614643030087;
    v = 3.770187233889873e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .4549287258889735;
    b = .1638770827382693;
    v = 3.878608613694378e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .5000278512957279;
    b = .2065965798260176;
    v = 3.959065270221274e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .5429785044928199;
    b = .2489436378852235;
    v = 4.01528697546357e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .5835939850491711;
    b = .2904811368946891;
    v = 4.050866785614717e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .6216870353444856;
    b = .3307941957666609;
    v = 4.069320185051913e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .4151104662709091;
    b = .04064829146052554;
    v = 3.760120964062763e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .4649804275009218;
    b = .08258424547294755;
    v = 3.870969564418064e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .5124695757009662;
    b = .1251841962027289;
    v = 3.955287790534055e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .5574711100606224;
    b = .1679107505976331;
    v = 4.015361911302668e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .5998597333287227;
    b = .2102805057358715;
    v = 4.053836986719548e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .63950071485166;
    b = .2518418087774107;
    v = 4.073578673299117e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .5188456224746252;
    b = .04194321676077518;
    v = 3.954628379231406e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .5664190707942778;
    b = .08457661551921499;
    v = 4.01764550884753e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .6110464353283153;
    b = .1273652932519396;
    v = 4.059030348651293e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .6526430302051563;
    b = .1698173239076354;
    v = 4.08056580948488e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .6167551880377548;
    b = .04266398851548864;
    v = 4.063018753664651e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .6607195418355383;
    b = .08551925814238349;
    v = 4.087191292799671e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    --(*n);
    return 0;
}

int ld3074_(double *x, double *y, double *z, double *w, int *n)
{
    static double a, b, v;

    /* VW */
    /* VW    LEBEDEV 3074-POINT ANGULAR GRID */
    /* VW */
    /* hvd */
    /* hvd   This subroutine is part of a set of subroutines that generate */
    /* hvd   Lebedev grids [1-6] for integration on a sphere. The original */
    /* hvd   C-code [1] was kindly provided by Dr. Dmitri N. Laikov and */
    /* hvd   translated into fortran by Dr. Christoph van Wuellen. */
    /* hvd   This subroutine was translated using a C to fortran77 conversion */
    /* hvd   tool written by Dr. Christoph van Wuellen. */
    /* hvd */
    /* hvd   Users of this code are asked to include reference [1] in their */
    /* hvd   publications, and in the user- and programmers-manuals */
    /* hvd   describing their codes. */
    /* hvd */
    /* hvd   This code was distributed through CCL (http://www.ccl.net/). */
    /* hvd */
    /* hvd   [1] V.I. Lebedev, and D.N. Laikov */
    /* hvd       "A quadrature formula for the sphere of the 131st */
    /* hvd        algebraic order of accuracy" */
    /* hvd       Doklady Mathematics, Vol. 59, No. 3, 1999, pp. 477-481. */
    /* hvd */
    /* hvd   [2] V.I. Lebedev */
    /* hvd       "A quadrature formula for the sphere of 59th algebraic */
    /* hvd        order of accuracy" */
    /* hvd       Russian Acad. Sci. Dokl. Math., Vol. 50, 1995, pp. 283-286. */
    /* hvd */
    /* hvd   [3] V.I. Lebedev, and A.L. Skorokhodov */
    /* hvd       "Quadrature formulas of orders 41, 47, and 53 for the sphere" */
    /* hvd       Russian Acad. Sci. Dokl. Math., Vol. 45, 1992, pp. 587-592. */
    /* hvd */
    /* hvd   [4] V.I. Lebedev */
    /* hvd       "Spherical quadrature formulas exact to orders 25-29" */
    /* hvd       Siberian Mathematical Journal, Vol. 18, 1977, pp. 99-107. */
    /* hvd */
    /* hvd   [5] V.I. Lebedev */
    /* hvd       "Quadratures on a sphere" */
    /* hvd       Computational Mathematics and Mathematical Physics, Vol. 16, */
    /* hvd       1976, pp. 10-24. */
    /* hvd */
    /* hvd   [6] V.I. Lebedev */
    /* hvd       "Values of the nodes and weights of ninth to seventeenth */
    /* hvd        order Gauss-Markov quadrature formulae invariant under the */
    /* hvd        octahedron group with inversion" */
    /* hvd       Computational Mathematics and Mathematical Physics, Vol. 15, */
    /* hvd       1975, pp. 44-51. */
    /* hvd */
    /* Parameter adjustments */
    --w;
    --z;
    --y;
    --x;

    /* Function Body */
    *n = 1;
    v = 2.599095953754734e-5;
    gen_oh_(1, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    v = 3.603134089687541e-4;
    gen_oh_(2, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    v = 3.586067974412447e-4;
    gen_oh_(3, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .01886108518723392;
    v = 9.83152847438588e-5;
    gen_oh_(4, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .04800217244625303;
    v = 1.60502310795445e-4;
    gen_oh_(4, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .08244922058397242;
    v = 2.072200131464099e-4;
    gen_oh_(4, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .1200408362484023;
    v = 2.431297618814187e-4;
    gen_oh_(4, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .1595773530809965;
    v = 2.711819064496707e-4;
    gen_oh_(4, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .2002635973434064;
    v = 2.932762038321116e-4;
    gen_oh_(4, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .2415127590139982;
    v = 3.107032514197368e-4;
    gen_oh_(4, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .2828584158458477;
    v = 3.243808058921213e-4;
    gen_oh_(4, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .3239091015338138;
    v = 3.34989909137403e-4;
    gen_oh_(4, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .3643225097962194;
    v = 3.430580688505218e-4;
    gen_oh_(4, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .4037897083691802;
    v = 3.490124109290343e-4;
    gen_oh_(4, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .4420247515194127;
    v = 3.532148948561955e-4;
    gen_oh_(4, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .4787572538464938;
    v = 3.559862669062833e-4;
    gen_oh_(4, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .5137265251275234;
    v = 3.576224317551411e-4;
    gen_oh_(4, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .5466764056654611;
    v = 3.584050533086076e-4;
    gen_oh_(4, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .6054859420813535;
    v = 3.584903581373224e-4;
    gen_oh_(4, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .6308106701764562;
    v = 3.582991879040586e-4;
    gen_oh_(4, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .6530369230179584;
    v = 3.582371187963125e-4;
    gen_oh_(4, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .6718609524611158;
    v = 3.58435363112235e-4;
    gen_oh_(4, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .6869676499894013;
    v = 3.589120166517785e-4;
    gen_oh_(4, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .6980467077240748;
    v = 3.595445704531601e-4;
    gen_oh_(4, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .7048241721250522;
    v = 3.600943557111074e-4;
    gen_oh_(4, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .05591105222058232;
    v = 1.456447096742039e-4;
    gen_oh_(5, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .1407384078513916;
    v = 2.252370188283782e-4;
    gen_oh_(5, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .2364035438976309;
    v = 2.766135443474897e-4;
    gen_oh_(5, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .336060273781817;
    v = 3.110729491500851e-4;
    gen_oh_(5, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .4356292630054665;
    v = 3.342506712303391e-4;
    gen_oh_(5, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .5321569415256174;
    v = 3.49198183402686e-4;
    gen_oh_(5, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .6232956305040554;
    v = 3.576003604348932e-4;
    gen_oh_(5, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .09469870086838469;
    b = .0277874838730947;
    v = 1.921921305788564e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .1353170300568141;
    b = .06076569878628364;
    v = 2.301458216495632e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .1771679481726077;
    b = .0970307276271104;
    v = 2.604248549522893e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .2197066664231751;
    b = .1354112458524762;
    v = 2.845275425870697e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .2624783557374927;
    b = .17509964797441;
    v = 3.03687089797484e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .3050969521214442;
    b = .2154896907449802;
    v = 3.188414832298066e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .3472252637196021;
    b = .2560954625740152;
    v = 3.307046414722089e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .388561021902636;
    b = .2965070050624096;
    v = 3.39833096903136e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .4288273776062765;
    b = .3363641488734497;
    v = 3.466757899705373e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .4677662471302948;
    b = .3753400029836788;
    v = 3.516095923230054e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .5051333589553359;
    b = .4131297522144286;
    v = 3.549645184048486e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .5406942145810492;
    b = .4494423776081795;
    v = 3.570415969441392e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .5742204122576457;
    b = .4839938958841502;
    v = 3.581251798496118e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .1865407027225188;
    b = .03259144851070796;
    v = 2.543491329913348e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .2321186453689432;
    b = .06835679505297343;
    v = 2.786711051330776e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .2773159142523882;
    b = .1062284864451989;
    v = 2.985552361083679e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .3219200192237254;
    b = .1454404409323047;
    v = 3.145867929154039e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .3657032593944029;
    b = .185401828258251;
    v = 3.273290662067609e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .4084376778363622;
    b = .225629741201475;
    v = 3.372705511943501e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .4499004945751427;
    b = .2657104425000896;
    v = 3.44827443785151e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .4898758141326335;
    b = .3052755487631557;
    v = 3.503592783048583e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .5281547442266309;
    b = .3439863920645423;
    v = 3.541854792663162e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .5645346989813992;
    b = .3815229456121914;
    v = 3.565995517909428e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .5988181252159848;
    b = .4175752420966734;
    v = 3.578802078302898e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .2850425424471603;
    b = .03562149509862536;
    v = 2.958644592860982e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .3324619433027876;
    b = .07330318886871096;
    v = 3.119548129116835e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .3785848333076282;
    b = .1123226296008472;
    v = 3.250745225005984e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .4232891028562115;
    b = .1521084193337708;
    v = 3.355153415935208e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .4664287050829722;
    b = .192184445922361;
    v = 3.435847568549328e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .5078458493735726;
    b = .2321360989678303;
    v = 3.495786831622488e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .547377981620418;
    b = .271588648636052;
    v = 3.537767805534621e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .5848617133811376;
    b = .3101924707571355;
    v = 3.564459815421428e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .6201348281584888;
    b = .3476121052890973;
    v = 3.578464061225468e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .3852191185387871;
    b = .03763224880035108;
    v = 3.239748762836212e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .4325025061073423;
    b = .07659581935637135;
    v = 3.345491784174287e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .477848622973449;
    b = .11633813060839;
    v = 3.429126177301782e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .5211663693009;
    b = .1563890598752899;
    v = 3.492420343097421e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .5623469504853703;
    b = .19633208101492;
    v = 3.537399050235257e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .6012718188659246;
    b = .2357847407258738;
    v = 3.566209152659172e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .6378179206390117;
    b = .274384612124406;
    v = 3.581084321919782e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .4836936460214534;
    b = .03895902610739024;
    v = 3.426522117591512e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .5293792562683797;
    b = .0787124681931264;
    v = 3.491848770121379e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .5726281253100033;
    b = .1187963808202981;
    v = 3.539318235231476e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .6133658776169068;
    b = .1587914708061787;
    v = 3.570231438458694e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .6515085491865307;
    b = .1983058575227646;
    v = 3.586207335051714e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .5778692716064976;
    b = .03977209689791542;
    v = 3.541196205164025e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .6207904288086192;
    b = .07990157592981152;
    v = 3.574296911573953e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .6608688171046802;
    b = .1199671308754309;
    v = 3.591993279818963e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .665626308948913;
    b = .04015955957805969;
    v = 3.595855034661997e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    --(*n);
    return 0;
}

int ld3470_(double *x, double *y, double *z, double *w, int *n)
{
    static double a, b, v;

    /* VW */
    /* VW    LEBEDEV 3470-POINT ANGULAR GRID */
    /* VW */
    /* hvd */
    /* hvd   This subroutine is part of a set of subroutines that generate */
    /* hvd   Lebedev grids [1-6] for integration on a sphere. The original */
    /* hvd   C-code [1] was kindly provided by Dr. Dmitri N. Laikov and */
    /* hvd   translated into fortran by Dr. Christoph van Wuellen. */
    /* hvd   This subroutine was translated using a C to fortran77 conversion */
    /* hvd   tool written by Dr. Christoph van Wuellen. */
    /* hvd */
    /* hvd   Users of this code are asked to include reference [1] in their */
    /* hvd   publications, and in the user- and programmers-manuals */
    /* hvd   describing their codes. */
    /* hvd */
    /* hvd   This code was distributed through CCL (http://www.ccl.net/). */
    /* hvd */
    /* hvd   [1] V.I. Lebedev, and D.N. Laikov */
    /* hvd       "A quadrature formula for the sphere of the 131st */
    /* hvd        algebraic order of accuracy" */
    /* hvd       Doklady Mathematics, Vol. 59, No. 3, 1999, pp. 477-481. */
    /* hvd */
    /* hvd   [2] V.I. Lebedev */
    /* hvd       "A quadrature formula for the sphere of 59th algebraic */
    /* hvd        order of accuracy" */
    /* hvd       Russian Acad. Sci. Dokl. Math., Vol. 50, 1995, pp. 283-286. */
    /* hvd */
    /* hvd   [3] V.I. Lebedev, and A.L. Skorokhodov */
    /* hvd       "Quadrature formulas of orders 41, 47, and 53 for the sphere" */
    /* hvd       Russian Acad. Sci. Dokl. Math., Vol. 45, 1992, pp. 587-592. */
    /* hvd */
    /* hvd   [4] V.I. Lebedev */
    /* hvd       "Spherical quadrature formulas exact to orders 25-29" */
    /* hvd       Siberian Mathematical Journal, Vol. 18, 1977, pp. 99-107. */
    /* hvd */
    /* hvd   [5] V.I. Lebedev */
    /* hvd       "Quadratures on a sphere" */
    /* hvd       Computational Mathematics and Mathematical Physics, Vol. 16, */
    /* hvd       1976, pp. 10-24. */
    /* hvd */
    /* hvd   [6] V.I. Lebedev */
    /* hvd       "Values of the nodes and weights of ninth to seventeenth */
    /* hvd        order Gauss-Markov quadrature formulae invariant under the */
    /* hvd        octahedron group with inversion" */
    /* hvd       Computational Mathematics and Mathematical Physics, Vol. 15, */
    /* hvd       1975, pp. 44-51. */
    /* hvd */
    /* Parameter adjustments */
    --w;
    --z;
    --y;
    --x;

    /* Function Body */
    *n = 1;
    v = 2.04038273082633e-5;
    gen_oh_(1, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    v = 3.178149703889544e-4;
    gen_oh_(3, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .01721420832906233;
    v = 8.28811512807611e-5;
    gen_oh_(4, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .0440887537498177;
    v = 1.360883192522954e-4;
    gen_oh_(4, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .07594680813878681;
    v = 1.766854454542662e-4;
    gen_oh_(4, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .1108335359204799;
    v = 2.083153161230153e-4;
    gen_oh_(4, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .1476517054388567;
    v = 2.333279544657158e-4;
    gen_oh_(4, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .1856731870860615;
    v = 2.532809539930247e-4;
    gen_oh_(4, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .2243634099428821;
    v = 2.692472184211158e-4;
    gen_oh_(4, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .2633006881662727;
    v = 2.819949946811885e-4;
    gen_oh_(4, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .3021340904916283;
    v = 2.92095359397303e-4;
    gen_oh_(4, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .3405594048030089;
    v = 2.999889782948352e-4;
    gen_oh_(4, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .3783044434007372;
    v = 3.060292120496902e-4;
    gen_oh_(4, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .415119476740791;
    v = 3.105109167522192e-4;
    gen_oh_(4, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .4507705766443257;
    v = 3.136902387550312e-4;
    gen_oh_(4, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .4850346056573187;
    v = 3.157984652454632e-4;
    gen_oh_(4, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .517695081779247;
    v = 3.170516518425422e-4;
    gen_oh_(4, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .5485384240820989;
    v = 3.176568425633755e-4;
    gen_oh_(4, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .6039117238943308;
    v = 3.177198411207062e-4;
    gen_oh_(4, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .6279956655573113;
    v = 3.175519492394733e-4;
    gen_oh_(4, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .6493636169568952;
    v = 3.174654952634756e-4;
    gen_oh_(4, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .6677644117704504;
    v = 3.175676415467654e-4;
    gen_oh_(4, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .6829368572115624;
    v = 3.17892341783541e-4;
    gen_oh_(4, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .6946195818184121;
    v = 3.183788287531909e-4;
    gen_oh_(4, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .7025711542057026;
    v = 3.188755151918807e-4;
    gen_oh_(4, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .7066004767140119;
    v = 3.191916889313849e-4;
    gen_oh_(4, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .05132537689946062;
    v = 1.231779611744508e-4;
    gen_oh_(5, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .1297994661331225;
    v = 1.92466137383988e-4;
    gen_oh_(5, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .2188852049401307;
    v = 2.380881867403424e-4;
    gen_oh_(5, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .3123174824903457;
    v = 2.693100663037885e-4;
    gen_oh_(5, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .4064037620738195;
    v = 2.908673382834366e-4;
    gen_oh_(5, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .4984958396944782;
    v = 3.053914619381535e-4;
    gen_oh_(5, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .5864975046021365;
    v = 3.143916684147777e-4;
    gen_oh_(5, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .6686711634580175;
    v = 3.187042244055363e-4;
    gen_oh_(5, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .0871573878083595;
    b = .02557175233367578;
    v = 1.63521953586979e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .1248383123134007;
    b = .05604823383376681;
    v = 1.96810991769607e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .1638062693383378;
    b = .08968568601900765;
    v = 2.236754342249974e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .2035586203373176;
    b = .1254086651976279;
    v = 2.453186687017181e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .2436798975293774;
    b = .1624780150162012;
    v = 2.627551791580541e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .2838207507773806;
    b = .2003422342683208;
    v = 2.76765486015222e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .3236787502217692;
    b = .2385628026255263;
    v = 2.879467027765895e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .3629849554840691;
    b = .2767731148783578;
    v = 2.967639918918702e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .4014948081992087;
    b = .3146542308245309;
    v = 3.035900684660351e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .4389818379260225;
    b = .3519196415895088;
    v = 3.087338237298308e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .4752331143674377;
    b = .3883050984023654;
    v = 3.124608838860167e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .5100457318374018;
    b = .4235613423908649;
    v = 3.150084294226743e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .5432238388954868;
    b = .457448471719622;
    v = 3.165958398598402e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .5745758685072442;
    b = .4897311639255524;
    v = 3.174320440957372e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .1723981437592809;
    b = .03010630597881105;
    v = 2.182188909812599e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .2149553257844597;
    b = .06326031554204694;
    v = 2.399727933921445e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .2573256081247422;
    b = .09848566980258631;
    v = 2.579796133514652e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .2993163751238106;
    b = .1350835952384266;
    v = 2.727114052623535e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .3407238005148;
    b = .1725184055442181;
    v = 2.846327656281355e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .3813454978483264;
    b = .2103559279730725;
    v = 2.941491102051334e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .4209848104423343;
    b = .248227877455486;
    v = 3.016049492136107e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .45945196999963;
    b = .2858099509982883;
    v = 3.072949726175648e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .496564016618593;
    b = .3228075659915428;
    v = 3.11476814288646e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .5321441655571562;
    b = .3589459907204151;
    v = 3.143823673666223e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .5660208438582166;
    b = .393963008886431;
    v = 3.162269764661535e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .5980264315964364;
    b = .4276029922949089;
    v = 3.172164663759821e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .2644215852350733;
    b = .03300939429072552;
    v = 2.554575398967435e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .3090113743443063;
    b = .06803887650078501;
    v = 2.701704069135677e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .3525871079197808;
    b = .1044326136206709;
    v = 2.82369341346894e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .3950418005354029;
    b = .1416751597517679;
    v = 2.922898463214289e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .4362475663430163;
    b = .1793408610504821;
    v = 3.001829062162428e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .4760661812145854;
    b = .2170630750175722;
    v = 3.062890864542953e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .5143551042512103;
    b = .2545145157815807;
    v = 3.108328279264746e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .5509709026935597;
    b = .2913940101706601;
    v = 3.140243146201245e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .5857711030329428;
    b = .3274169910910705;
    v = 3.16063803097713e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .6186149917404392;
    b = .3623081329317265;
    v = 3.171462882206275e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .3586894569557064;
    b = .0349735438645004;
    v = 2.812388416031796e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .4035266610019441;
    b = .07129736739757095;
    v = 2.912137500288045e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .446777531233251;
    b = .1084758620193165;
    v = 2.993241256502206e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .4883638346608543;
    b = .1460915689241772;
    v = 3.057101738983822e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .5281908348434601;
    b = .183779083236998;
    v = 3.105319326251432e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .5661542687149311;
    b = .2212075390874021;
    v = 3.139565514428167e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .6021450102031452;
    b = .2580682841160985;
    v = 3.161543006806366e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .636052078361005;
    b = .2940656362094121;
    v = 3.172985960613294e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .4521611065087196;
    b = .03631055365867002;
    v = 2.989400336901431e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .4959365651560963;
    b = .0734831846848435;
    v = 3.054555883947677e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .5376815804038283;
    b = .1111087643812648;
    v = 3.104764960807702e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .5773314480243768;
    b = .1488226085145408;
    v = 3.141015825977616e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .6148113245575056;
    b = .1862892274135151;
    v = 3.164520621159896e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .650040746284238;
    b = .2231909701714456;
    v = 3.176652305912204e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .5425151448707213;
    b = .03718201306118944;
    v = 3.105097161023939e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .5841860556907931;
    b = .07483616335067346;
    v = 3.14301411789055e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .62346321868515;
    b = .112599083426612;
    v = 3.1681728662872e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .6602934551848843;
    b = .1501303813157619;
    v = 3.181401865570968e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .6278573968375105;
    b = .0376755993024572;
    v = 3.170663659156037e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .6665611711264577;
    b = .07548443301360158;
    v = 3.18544794462551e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    --(*n);
    return 0;
}

int ld3890_(double *x, double *y, double *z, double *w, int *n)
{
    static double a, b, v;

    /* VW */
    /* VW    LEBEDEV 3890-POINT ANGULAR GRID */
    /* VW */
    /* hvd */
    /* hvd   This subroutine is part of a set of subroutines that generate */
    /* hvd   Lebedev grids [1-6] for integration on a sphere. The original */
    /* hvd   C-code [1] was kindly provided by Dr. Dmitri N. Laikov and */
    /* hvd   translated into fortran by Dr. Christoph van Wuellen. */
    /* hvd   This subroutine was translated using a C to fortran77 conversion */
    /* hvd   tool written by Dr. Christoph van Wuellen. */
    /* hvd */
    /* hvd   Users of this code are asked to include reference [1] in their */
    /* hvd   publications, and in the user- and programmers-manuals */
    /* hvd   describing their codes. */
    /* hvd */
    /* hvd   This code was distributed through CCL (http://www.ccl.net/). */
    /* hvd */
    /* hvd   [1] V.I. Lebedev, and D.N. Laikov */
    /* hvd       "A quadrature formula for the sphere of the 131st */
    /* hvd        algebraic order of accuracy" */
    /* hvd       Doklady Mathematics, Vol. 59, No. 3, 1999, pp. 477-481. */
    /* hvd */
    /* hvd   [2] V.I. Lebedev */
    /* hvd       "A quadrature formula for the sphere of 59th algebraic */
    /* hvd        order of accuracy" */
    /* hvd       Russian Acad. Sci. Dokl. Math., Vol. 50, 1995, pp. 283-286. */
    /* hvd */
    /* hvd   [3] V.I. Lebedev, and A.L. Skorokhodov */
    /* hvd       "Quadrature formulas of orders 41, 47, and 53 for the sphere" */
    /* hvd       Russian Acad. Sci. Dokl. Math., Vol. 45, 1992, pp. 587-592. */
    /* hvd */
    /* hvd   [4] V.I. Lebedev */
    /* hvd       "Spherical quadrature formulas exact to orders 25-29" */
    /* hvd       Siberian Mathematical Journal, Vol. 18, 1977, pp. 99-107. */
    /* hvd */
    /* hvd   [5] V.I. Lebedev */
    /* hvd       "Quadratures on a sphere" */
    /* hvd       Computational Mathematics and Mathematical Physics, Vol. 16, */
    /* hvd       1976, pp. 10-24. */
    /* hvd */
    /* hvd   [6] V.I. Lebedev */
    /* hvd       "Values of the nodes and weights of ninth to seventeenth */
    /* hvd        order Gauss-Markov quadrature formulae invariant under the */
    /* hvd        octahedron group with inversion" */
    /* hvd       Computational Mathematics and Mathematical Physics, Vol. 15, */
    /* hvd       1975, pp. 44-51. */
    /* hvd */
    /* Parameter adjustments */
    --w;
    --z;
    --y;
    --x;

    /* Function Body */
    *n = 1;
    v = 1.80739525219692e-5;
    gen_oh_(1, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    v = 2.848008782238827e-4;
    gen_oh_(2, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    v = 2.836065837530581e-4;
    gen_oh_(3, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .01587876419858352;
    v = 7.013149266673816e-5;
    gen_oh_(4, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .04069193593751206;
    v = 1.162798021956766e-4;
    gen_oh_(4, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .07025888115257997;
    v = 1.518728583972105e-4;
    gen_oh_(4, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .1027495450028704;
    v = 1.798796108216934e-4;
    gen_oh_(4, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .1371457730893426;
    v = 2.022593385972785e-4;
    gen_oh_(4, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .1727758532671953;
    v = 2.203093105575464e-4;
    gen_oh_(4, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .2091492038929037;
    v = 2.349294234299855e-4;
    gen_oh_(4, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .2458813281751915;
    v = 2.467682058747003e-4;
    gen_oh_(4, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .2826545859450066;
    v = 2.563092683572224e-4;
    gen_oh_(4, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .3191957291799622;
    v = 2.639253896763318e-4;
    gen_oh_(4, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .3552621469299578;
    v = 2.699137479265108e-4;
    gen_oh_(4, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .390632950340623;
    v = 2.745196420166739e-4;
    gen_oh_(4, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .4251028614093031;
    v = 2.779529197397593e-4;
    gen_oh_(4, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .458477752011187;
    v = 2.803996086684265e-4;
    gen_oh_(4, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .4905711358710193;
    v = 2.820302356715842e-4;
    gen_oh_(4, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .5212011669847385;
    v = 2.830056747491068e-4;
    gen_oh_(4, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .5501878488737995;
    v = 2.834808950776839e-4;
    gen_oh_(4, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .6025037877479342;
    v = 2.835282339078929e-4;
    gen_oh_(4, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .6254572689549016;
    v = 2.8338192670658e-4;
    gen_oh_(4, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .6460107179528248;
    v = 2.832858336906784e-4;
    gen_oh_(4, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .6639541138154251;
    v = 2.833268235451244e-4;
    gen_oh_(4, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .6790688515667495;
    v = 2.835432677029253e-4;
    gen_oh_(4, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .6911338580371512;
    v = 2.839091722743049e-4;
    gen_oh_(4, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .699938595612649;
    v = 2.843308178875841e-4;
    gen_oh_(4, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .7053037748656896;
    v = 2.846703550533846e-4;
    gen_oh_(4, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .04732224387180115;
    v = 1.0511934069719e-4;
    gen_oh_(5, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .1202100529326803;
    v = 1.657871838796974e-4;
    gen_oh_(5, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .2034304820664855;
    v = 2.064648113714232e-4;
    gen_oh_(5, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .2912285643573002;
    v = 2.347942745819741e-4;
    gen_oh_(5, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .3802361792726768;
    v = 2.547775326597726e-4;
    gen_oh_(5, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .4680598511056146;
    v = 2.686876684847025e-4;
    gen_oh_(5, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .5528151052155599;
    v = 2.778665755515867e-4;
    gen_oh_(5, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .6329386307803041;
    v = 2.830996616782929e-4;
    gen_oh_(5, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .08056516651369069;
    b = .02363454684003124;
    v = 1.403063340168372e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .1156476077139389;
    b = .05191291632545936;
    v = 1.696504125939477e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .1520473382760421;
    b = .08322715736994519;
    v = 1.93578724274539e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .1892986699745931;
    b = .1165855667993712;
    v = 2.130614510521968e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .2270194446777792;
    b = .1513077167409504;
    v = 2.289381265931048e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .2648908185093273;
    b = .1868882025807859;
    v = 2.418630292816186e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .3026389259574136;
    b = .2229277629776224;
    v = 2.523400495631193e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .3400220296151384;
    b = .2590951840746235;
    v = 2.607623973449605e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .376821795333551;
    b = .2951047291750847;
    v = 2.674441032689209e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .4128372900921884;
    b = .330701971416993;
    v = 2.726432360343356e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .447880713181563;
    b = .3656544101087634;
    v = 2.765787685924545e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .4817742034089257;
    b = .3997448951939695;
    v = 2.794428690642224e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .5143472814653344;
    b = .4327667110812024;
    v = 2.814099002062895e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .545434621390565;
    b = .4645196123532293;
    v = 2.826429531578994e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .5748739313170252;
    b = .4948063555703345;
    v = 2.832983542550884e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .1599598738286342;
    b = .02792357590048985;
    v = 1.886695565284976e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .1998097412500951;
    b = .05877141038139065;
    v = 2.081867882748234e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .2396228952566202;
    b = .09164573914691377;
    v = 2.245148680600796e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .2792228341097746;
    b = .1259049641962687;
    v = 2.380370491511872e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .3184251107546741;
    b = .1610594823400863;
    v = 2.491398041852455e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .3570481164426244;
    b = .1967151653460898;
    v = 2.58163240588123e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .3949164710492144;
    b = .2325404606175168;
    v = 2.653965506227417e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .4318617293970503;
    b = .2682461141151439;
    v = 2.710857216747087e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .4677221009931678;
    b = .3035720116011973;
    v = 2.754434093903659e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .5023417939270955;
    b = .3382781859197439;
    v = 2.78657993251938e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .5355701836636128;
    b = .3721383065625942;
    v = 2.809011080679474e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .5672608451328771;
    b = .4049346360466055;
    v = 2.823336184560987e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .5972704202540162;
    b = .4364538098633802;
    v = 2.831101175806309e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .2461687022333596;
    b = .03070423166833368;
    v = 2.221679970354546e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .2881774566286831;
    b = .06338034669281885;
    v = 2.356185734270703e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .3293963604116978;
    b = .09742862487067941;
    v = 2.46922834480559e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .3697303822241377;
    b = .132379953228229;
    v = 2.562726348642046e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .4090663023135127;
    b = .1678497018129336;
    v = 2.638756726753028e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .4472819355411712;
    b = .2035095105326114;
    v = 2.699311157390862e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .4842513377231437;
    b = .2390692566672091;
    v = 2.746233268403837e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .5198477629962928;
    b = .2742649818076149;
    v = 2.781225674454771e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .5539453011883145;
    b = .3088503806580094;
    v = 2.805881254045684e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .5864196762401251;
    b = .3425904245906614;
    v = 2.821719877004913e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .617148446666839;
    b = .3752562294789468;
    v = 2.830222502333124e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .3350337830565727;
    b = .03261589934634747;
    v = 2.45799595674487e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .3775773224758284;
    b = .06658438928081572;
    v = 2.551474407503706e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .4188155229848973;
    b = .1014565797157954;
    v = 2.629065335195311e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .4586805892009344;
    b = .1368573320843822;
    v = 2.691900449925075e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .4970895714224235;
    b = .1724614851951608;
    v = 2.741275485754276e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .5339505133960747;
    b = .2079779381416412;
    v = 2.778530970122595e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .569166579253144;
    b = .2431385788322288;
    v = 2.805010567646741e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .6026387682680377;
    b = .2776901883049853;
    v = 2.82205583403104e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .6342676150163307;
    b = .3113881356386632;
    v = 2.831016901243473e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .4237951119537067;
    b = .03394877848664351;
    v = 2.624474901131803e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .4656918683234929;
    b = .06880219556291447;
    v = 2.688034163039377e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .505885706918598;
    b = .1041946859721635;
    v = 2.738932751287636e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .5443204666713996;
    b = .1398039738736393;
    v = 2.777944791242523e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .5809298813759742;
    b = .1753373381196155;
    v = 2.806011661660987e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .6156416039447128;
    b = .210521579351401;
    v = 2.82418145659746e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .6483801351066604;
    b = .2450953312157051;
    v = 2.833585216577828e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .5103616577251688;
    b = .03485560643800719;
    v = 2.738165236962878e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .5506738792580681;
    b = .07026308631512033;
    v = 2.77836520820318e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .5889573040995292;
    b = .1059035061296403;
    v = 2.807852940418966e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .625164158951693;
    b = .1414823925236026;
    v = 2.827245949674705e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .6592414921570178;
    b = .176720790821453;
    v = 2.837342344829828e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .5930314017533384;
    b = .03542189339561672;
    v = 2.809233907610981e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .6309812253390175;
    b = .07109574040369549;
    v = 2.829930809742694e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .666629601135323;
    b = .106725979228273;
    v = 2.841097874111479e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .6703715271049922;
    b = .03569455268820809;
    v = 2.843455206008783e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    --(*n);
    return 0;
}

int ld4334_(double *x, double *y, double *z, double *w, int *n)
{
    static double a, b, v;

    /* VW */
    /* VW    LEBEDEV 4334-POINT ANGULAR GRID */
    /* VW */
    /* hvd */
    /* hvd   This subroutine is part of a set of subroutines that generate */
    /* hvd   Lebedev grids [1-6] for integration on a sphere. The original */
    /* hvd   C-code [1] was kindly provided by Dr. Dmitri N. Laikov and */
    /* hvd   translated into fortran by Dr. Christoph van Wuellen. */
    /* hvd   This subroutine was translated using a C to fortran77 conversion */
    /* hvd   tool written by Dr. Christoph van Wuellen. */
    /* hvd */
    /* hvd   Users of this code are asked to include reference [1] in their */
    /* hvd   publications, and in the user- and programmers-manuals */
    /* hvd   describing their codes. */
    /* hvd */
    /* hvd   This code was distributed through CCL (http://www.ccl.net/). */
    /* hvd */
    /* hvd   [1] V.I. Lebedev, and D.N. Laikov */
    /* hvd       "A quadrature formula for the sphere of the 131st */
    /* hvd        algebraic order of accuracy" */
    /* hvd       Doklady Mathematics, Vol. 59, No. 3, 1999, pp. 477-481. */
    /* hvd */
    /* hvd   [2] V.I. Lebedev */
    /* hvd       "A quadrature formula for the sphere of 59th algebraic */
    /* hvd        order of accuracy" */
    /* hvd       Russian Acad. Sci. Dokl. Math., Vol. 50, 1995, pp. 283-286. */
    /* hvd */
    /* hvd   [3] V.I. Lebedev, and A.L. Skorokhodov */
    /* hvd       "Quadrature formulas of orders 41, 47, and 53 for the sphere" */
    /* hvd       Russian Acad. Sci. Dokl. Math., Vol. 45, 1992, pp. 587-592. */
    /* hvd */
    /* hvd   [4] V.I. Lebedev */
    /* hvd       "Spherical quadrature formulas exact to orders 25-29" */
    /* hvd       Siberian Mathematical Journal, Vol. 18, 1977, pp. 99-107. */
    /* hvd */
    /* hvd   [5] V.I. Lebedev */
    /* hvd       "Quadratures on a sphere" */
    /* hvd       Computational Mathematics and Mathematical Physics, Vol. 16, */
    /* hvd       1976, pp. 10-24. */
    /* hvd */
    /* hvd   [6] V.I. Lebedev */
    /* hvd       "Values of the nodes and weights of ninth to seventeenth */
    /* hvd        order Gauss-Markov quadrature formulae invariant under the */
    /* hvd        octahedron group with inversion" */
    /* hvd       Computational Mathematics and Mathematical Physics, Vol. 15, */
    /* hvd       1975, pp. 44-51. */
    /* hvd */
    /* Parameter adjustments */
    --w;
    --z;
    --y;
    --x;

    /* Function Body */
    *n = 1;
    v = 1.449063022537883e-5;
    gen_oh_(1, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    v = 2.546377329828424e-4;
    gen_oh_(3, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .01462896151831013;
    v = 6.018432961087496e-5;
    gen_oh_(4, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .03769840812493139;
    v = 1.002286583263673e-4;
    gen_oh_(4, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .06524701904096891;
    v = 1.315222931028093e-4;
    gen_oh_(4, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .09560543416134648;
    v = 1.564213746876724e-4;
    gen_oh_(4, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .1278335898929198;
    v = 1.765118841507736e-4;
    gen_oh_(4, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .1613096104466031;
    v = 1.92873709931108e-4;
    gen_oh_(4, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .1955806225745371;
    v = 2.06265853426327e-4;
    gen_oh_(4, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .2302935218498028;
    v = 2.172395445953787e-4;
    gen_oh_(4, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .2651584344113027;
    v = 2.262076188876047e-4;
    gen_oh_(4, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .2999276825183209;
    v = 2.334885699462397e-4;
    gen_oh_(4, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .3343828669718798;
    v = 2.393355273179203e-4;
    gen_oh_(4, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .3683265013750518;
    v = 2.439559200468863e-4;
    gen_oh_(4, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .4015763206518108;
    v = 2.475251866060002e-4;
    gen_oh_(4, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .433961202639977;
    v = 2.501965558158773e-4;
    gen_oh_(4, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .4653180651114582;
    v = 2.521081407925925e-4;
    gen_oh_(4, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .4954893331080803;
    v = 2.533881002388081e-4;
    gen_oh_(4, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .524320706892493;
    v = 2.541582900848261e-4;
    gen_oh_(4, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .5516590479041704;
    v = 2.54536573752586e-4;
    gen_oh_(4, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .6012371927804176;
    v = 2.545726993066799e-4;
    gen_oh_(4, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .6231574466449819;
    v = 2.544456197465555e-4;
    gen_oh_(4, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .6429416514181271;
    v = 2.543481596881064e-4;
    gen_oh_(4, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .6604124272943595;
    v = 2.543506451429194e-4;
    gen_oh_(4, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .675385147040825;
    v = 2.544905675493763e-4;
    gen_oh_(4, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .687671797062616;
    v = 2.547611407344429e-4;
    gen_oh_(4, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .6970895061319234;
    v = 2.551060375448869e-4;
    gen_oh_(4, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .703474691255331;
    v = 2.554291933816039e-4;
    gen_oh_(4, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .7067017217542295;
    v = 2.556255710686343e-4;
    gen_oh_(4, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .04382223501131123;
    v = 9.041339695118195e-5;
    gen_oh_(5, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .1117474077400006;
    v = 1.438426330079022e-4;
    gen_oh_(5, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .189715325291144;
    v = 1.802523089820518e-4;
    gen_oh_(5, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .2724023009910331;
    v = 2.060052290565496e-4;
    gen_oh_(5, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .3567163308709902;
    v = 2.245002248967466e-4;
    gen_oh_(5, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .4404784483028087;
    v = 2.37705984773115e-4;
    gen_oh_(5, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .5219833154161411;
    v = 2.468118955882525e-4;
    gen_oh_(5, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .5998179868977553;
    v = 2.525410872966528e-4;
    gen_oh_(5, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .6727803154548222;
    v = 2.553101409933397e-4;
    gen_oh_(5, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .07476563943166086;
    b = .02193168509461185;
    v = 1.212879733668632e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .1075341482001416;
    b = .04826419281533887;
    v = 1.472872881270931e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .1416344885203259;
    b = .07751191883575742;
    v = 1.686846601010828e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .1766325315388586;
    b = .108755813924768;
    v = 1.862698414660208e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .2121744174481514;
    b = .1413661374253096;
    v = 2.007430956991861e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .2479669443408145;
    b = .174876821425888;
    v = 2.126568125394796e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .2837600452294113;
    b = .2089216406612073;
    v = 2.224394603372113e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .3193344933193984;
    b = .2431987685545972;
    v = 2.304264522673135e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .3544935442438745;
    b = .277449705437777;
    v = 2.368854288424087e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .3890571932288154;
    b = .3114460356156915;
    v = 2.420352089461772e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .422858121425909;
    b = .3449806851913012;
    v = 2.460597113081295e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .4557387211304052;
    b = .3778618641248256;
    v = 2.491181912257687e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .4875487950541643;
    b = .4099086391698978;
    v = 2.513528194205857e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .5181436529962997;
    b = .4409474925853973;
    v = 2.52894309669322e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .5473824095600661;
    b = .4708094517711291;
    v = 2.538660368488136e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .5751263398976174;
    b = .4993275140354637;
    v = 2.543868648299022e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .1489515746840028;
    b = .02599381993267017;
    v = 1.642595537825183e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .1863656444351767;
    b = .0547928653246219;
    v = 1.818246659849308e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .2238602880356348;
    b = .08556763251425254;
    v = 1.96656564949242e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .261272337572816;
    b = .1177257802267011;
    v = 2.090677905657991e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .298433299020619;
    b = .15081684561927;
    v = 2.193820409510504e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .3351786584663333;
    b = .1844801892177727;
    v = 2.278870827661928e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .371350552220912;
    b = .2184145236087598;
    v = 2.34828319228209e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .4067981098954663;
    b = .2523590641486229;
    v = 2.404139755581477e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .4413769993687534;
    b = .2860812976901373;
    v = 2.448227407760734e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .4749487182516394;
    b = .3193686757808996;
    v = 2.482110455592573e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .5073798105075426;
    b = .3520226949547602;
    v = 2.507192397774103e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .5385410448878654;
    b = .383854439566789;
    v = 2.52476596853488e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .568306535367053;
    b = .4146810037640963;
    v = 2.536052388539425e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .596552762066351;
    b = .4443224094681121;
    v = 2.542230588033068e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .2299227700856157;
    b = .02865757664057584;
    v = 1.944817013047896e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .2695752998553267;
    b = .05923421684485993;
    v = 2.067862362746635e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .3086178716611389;
    b = .09117817776057715;
    v = 2.172440734649114e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .3469649871659077;
    b = .1240593814082605;
    v = 2.260125991723423e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .3845153566319655;
    b = .1575272058259175;
    v = 2.332655008689523e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .4211600033403215;
    b = .1912845163525413;
    v = 2.391699681532458e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .4567867834329882;
    b = .2250710177858171;
    v = 2.438801528273928e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .4912829319232061;
    b = .258652130344091;
    v = 2.475370504260665e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .5245364793303812;
    b = .2918112242865407;
    v = 2.502707235640574e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .5564369788915756;
    b = .324343923906789;
    v = 2.522031701054241e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .5868757697775287;
    b = .3560536787835351;
    v = 2.534511269978784e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .6157458853519617;
    b = .3867480821242581;
    v = 2.541284914955151e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .3138461110672113;
    b = .03051374637507278;
    v = 2.161509250688394e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .3542495872050569;
    b = .06237111233730755;
    v = 2.248778513437852e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .3935751553120181;
    b = .09516223952401907;
    v = 2.322388803404617e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .4317634668111147;
    b = .1285467341508517;
    v = 2.383265471001355e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .4687413842250821;
    b = .1622318931656033;
    v = 2.432476675019525e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .5044274237060283;
    b = .1959581153836453;
    v = 2.471122223750674e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .5387354077925727;
    b = .2294888081183837;
    v = 2.50029175248687e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .5715768898356105;
    b = .2626031152713945;
    v = 2.521055942764682e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .6028627200136111;
    b = .2950904075286713;
    v = 2.534472785575503e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .6325039812653463;
    b = .3267458451113286;
    v = 2.541599713080121e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .3981986708423407;
    b = .03183291458749821;
    v = 2.317380975862936e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .43827911821333;
    b = .06459548193880908;
    v = 2.378550733719775e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .4769233057218166;
    b = .09795757037087952;
    v = 2.428884456739118e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .5140823911194238;
    b = .1316307235126655;
    v = 2.469002655757292e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .5496977833862983;
    b = .1653556486358704;
    v = 2.499657574265851e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .5837047306512727;
    b = .198893172412651;
    v = 2.521676168486082e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .6160349566926879;
    b = .232017458143895;
    v = 2.535935662645334e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .646618535320944;
    b = .2645106562168662;
    v = 2.543356743363214e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .4810835158795404;
    b = .03275917807743992;
    v = 2.427353285201535e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .5199925041324341;
    b = .06612546183967181;
    v = 2.468258039744386e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .5571717692207494;
    b = .09981498331474143;
    v = 2.50006095644031e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .5925789250836378;
    b = .1335687001410374;
    v = 2.523238365420979e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .626165852385967;
    b = .1671444402896463;
    v = 2.538399260252846e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .6578811126669331;
    b = .2003106382156076;
    v = 2.546255927268069e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .56096246129981;
    b = .03337500940231335;
    v = 2.500583360048449e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .597995965998467;
    b = .06708750335901803;
    v = 2.524777638260203e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .6330523711054002;
    b = .100879212642485;
    v = 2.540951193860656e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .6660960998103972;
    b = .1345050343171794;
    v = 2.549524085027472e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .6365384364585819;
    b = .03372799460737052;
    v = 2.542569507009158e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .6710994302899275;
    b = .06755249309678028;
    v = 2.552114127580376e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    --(*n);
    return 0;
}

int ld4802_(double *x, double *y, double *z, double *w, int *n)
{
    static double a, b, v;

    /* VW */
    /* VW    LEBEDEV 4802-POINT ANGULAR GRID */
    /* VW */
    /* hvd */
    /* hvd   This subroutine is part of a set of subroutines that generate */
    /* hvd   Lebedev grids [1-6] for integration on a sphere. The original */
    /* hvd   C-code [1] was kindly provided by Dr. Dmitri N. Laikov and */
    /* hvd   translated into fortran by Dr. Christoph van Wuellen. */
    /* hvd   This subroutine was translated using a C to fortran77 conversion */
    /* hvd   tool written by Dr. Christoph van Wuellen. */
    /* hvd */
    /* hvd   Users of this code are asked to include reference [1] in their */
    /* hvd   publications, and in the user- and programmers-manuals */
    /* hvd   describing their codes. */
    /* hvd */
    /* hvd   This code was distributed through CCL (http://www.ccl.net/). */
    /* hvd */
    /* hvd   [1] V.I. Lebedev, and D.N. Laikov */
    /* hvd       "A quadrature formula for the sphere of the 131st */
    /* hvd        algebraic order of accuracy" */
    /* hvd       Doklady Mathematics, Vol. 59, No. 3, 1999, pp. 477-481. */
    /* hvd */
    /* hvd   [2] V.I. Lebedev */
    /* hvd       "A quadrature formula for the sphere of 59th algebraic */
    /* hvd        order of accuracy" */
    /* hvd       Russian Acad. Sci. Dokl. Math., Vol. 50, 1995, pp. 283-286. */
    /* hvd */
    /* hvd   [3] V.I. Lebedev, and A.L. Skorokhodov */
    /* hvd       "Quadrature formulas of orders 41, 47, and 53 for the sphere" */
    /* hvd       Russian Acad. Sci. Dokl. Math., Vol. 45, 1992, pp. 587-592. */
    /* hvd */
    /* hvd   [4] V.I. Lebedev */
    /* hvd       "Spherical quadrature formulas exact to orders 25-29" */
    /* hvd       Siberian Mathematical Journal, Vol. 18, 1977, pp. 99-107. */
    /* hvd */
    /* hvd   [5] V.I. Lebedev */
    /* hvd       "Quadratures on a sphere" */
    /* hvd       Computational Mathematics and Mathematical Physics, Vol. 16, */
    /* hvd       1976, pp. 10-24. */
    /* hvd */
    /* hvd   [6] V.I. Lebedev */
    /* hvd       "Values of the nodes and weights of ninth to seventeenth */
    /* hvd        order Gauss-Markov quadrature formulae invariant under the */
    /* hvd        octahedron group with inversion" */
    /* hvd       Computational Mathematics and Mathematical Physics, Vol. 15, */
    /* hvd       1975, pp. 44-51. */
    /* hvd */
    /* Parameter adjustments */
    --w;
    --z;
    --y;
    --x;

    /* Function Body */
    *n = 1;
    v = 9.687521879420705e-5;
    gen_oh_(1, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    v = 2.307897895367918e-4;
    gen_oh_(2, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    v = 2.297310852498558e-4;
    gen_oh_(3, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .02335728608887064;
    v = 7.386265944001919e-5;
    gen_oh_(4, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .04352987836550653;
    v = 8.25797769854221e-5;
    gen_oh_(4, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .06439200521088801;
    v = 9.70604476205763e-5;
    gen_oh_(4, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .09003943631993181;
    v = 1.302393847117003e-4;
    gen_oh_(4, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .1196706615548473;
    v = 1.541957004600968e-4;
    gen_oh_(4, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .1511715412838134;
    v = 1.704459770092199e-4;
    gen_oh_(4, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .1835982828503801;
    v = 1.827374890942906e-4;
    gen_oh_(4, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .2165081259155405;
    v = 1.926360817436107e-4;
    gen_oh_(4, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .2496208720417563;
    v = 2.008010239494833e-4;
    gen_oh_(4, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .28272006735679;
    v = 2.075635983209175e-4;
    gen_oh_(4, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .3156190823994346;
    v = 2.131306638690909e-4;
    gen_oh_(4, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .3481476793749115;
    v = 2.176562329937335e-4;
    gen_oh_(4, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .3801466086947226;
    v = 2.212682262991018e-4;
    gen_oh_(4, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .4114652119634011;
    v = 2.240799515668565e-4;
    gen_oh_(4, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .4419598786519751;
    v = 2.261959816187525e-4;
    gen_oh_(4, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .4714925949329543;
    v = 2.277156368808855e-4;
    gen_oh_(4, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .4999293972879466;
    v = 2.287351772128336e-4;
    gen_oh_(4, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .5271387221431248;
    v = 2.293490814084085e-4;
    gen_oh_(4, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .5529896780837761;
    v = 2.296505312376273e-4;
    gen_oh_(4, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .6000856099481712;
    v = 2.296793832318756e-4;
    gen_oh_(4, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .6210562192785175;
    v = 2.295785443842974e-4;
    gen_oh_(4, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .640116587993424;
    v = 2.295017931529102e-4;
    gen_oh_(4, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .6571144029244334;
    v = 2.295059638184868e-4;
    gen_oh_(4, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .6718910821718863;
    v = 2.296232343237362e-4;
    gen_oh_(4, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .684284559109901;
    v = 2.298530178740771e-4;
    gen_oh_(4, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .6941353476269816;
    v = 2.301579790280501e-4;
    gen_oh_(4, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .7012965242212991;
    v = 2.304690404996513e-4;
    gen_oh_(4, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .7056471428242644;
    v = 2.307027995907102e-4;
    gen_oh_(4, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .04595557643585895;
    v = 9.312274696671092e-5;
    gen_oh_(5, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .1049316742435023;
    v = 1.199919385876926e-4;
    gen_oh_(5, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .1773548879549274;
    v = 1.59803913887769e-4;
    gen_oh_(5, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .2559071411236127;
    v = 1.8222537635749e-4;
    gen_oh_(5, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .3358156837985898;
    v = 1.98857959365504e-4;
    gen_oh_(5, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .4155835743763893;
    v = 2.112620102533307e-4;
    gen_oh_(5, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .4937894296167472;
    v = 2.201594887699007e-4;
    gen_oh_(5, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .5691569694793316;
    v = 2.261622590895036e-4;
    gen_oh_(5, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .6405840854894251;
    v = 2.296458453435705e-4;
    gen_oh_(5, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .07345133894143348;
    b = .02177844081486067;
    v = 1.006006990267e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .1009859834044931;
    b = .04590362185775188;
    v = 1.227676689635876e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .1324289619748758;
    b = .07255063095690877;
    v = 1.467864280270117e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .1654272109607127;
    b = .1017825451960684;
    v = 1.644178912101232e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .1990767186776461;
    b = .1325652320980364;
    v = 1.777664890718961e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .2330125945523278;
    b = .1642765374496765;
    v = 1.88482566451669e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .2670080611108287;
    b = .1965360374337889;
    v = 1.973269246453848e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .3008753376294316;
    b = .2290726770542238;
    v = 2.046767775855328e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .334447559616786;
    b = .2616645495370823;
    v = 2.10760012591804e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .3675709724070786;
    b = .2941150728843141;
    v = 2.157416362266829e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .4001000887587812;
    b = .3262440400919066;
    v = 2.197557816920721e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .4318956350436028;
    b = .3578835350611916;
    v = 2.229192611835437e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .4628239056795531;
    b = .3888751854043678;
    v = 2.253385110212775e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .4927563229773636;
    b = .419067800322284;
    v = 2.271137107548774e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .5215687136707969;
    b = .4483151836883852;
    v = 2.283414092917525e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .5491402346984905;
    b = .476474067608788;
    v = 2.291161673130077e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .5753520160126075;
    b = .5034021310998277;
    v = 2.295313908576598e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .1388326356417754;
    b = .02435436510372806;
    v = 1.438204721359031e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .1743686900537244;
    b = .05118897057342652;
    v = 1.607738025495257e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .2099737037950268;
    b = .08014695048539634;
    v = 1.741483853528379e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .2454492590908548;
    b = .1105117874155699;
    v = 1.851918467519151e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .2807219257864278;
    b = .1417950531570966;
    v = 1.944628638070613e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .3156842271975842;
    b = .1736604945719597;
    v = 2.022495446275152e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .3502090945177752;
    b = .2058466324693981;
    v = 2.087462382438514e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .3841684849519686;
    b = .2381284261195919;
    v = 2.141074754818308e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .4174372367906016;
    b = .2703031270422569;
    v = 2.184640913748162e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .4498926465011892;
    b = .3021845683091309;
    v = 2.219309165220329e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .4814146229807701;
    b = .333599335516572;
    v = 2.246123118340624e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .5118863625734701;
    b = .3643833735518232;
    v = 2.266062766915125e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .5411947455119144;
    b = .3943789541958179;
    v = 2.280072952230796e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .5692301500357246;
    b = .4234320144403542;
    v = 2.289082025202583e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .5958857204139576;
    b = .451389794741926;
    v = 2.294012695120025e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .2156270284785766;
    b = .02681225755444491;
    v = 1.722434488736947e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .253238505490971;
    b = .05557495747805614;
    v = 1.830237421455091e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .2902564617771537;
    b = .08569368062950249;
    v = 1.923855349997633e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .3266979823143256;
    b = .1167367450324135;
    v = 2.004067861936271e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .3625039627493614;
    b = .1483861994003304;
    v = 2.071817297354263e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .3975838937548699;
    b = .1803821503011405;
    v = 2.128250834102103e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .4318396099009774;
    b = .2124962965666424;
    v = 2.174513719440102e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .4651706555732742;
    b = .2445221837805913;
    v = 2.211661839150214e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .4974752649620969;
    b = .2762701224322987;
    v = 2.240665257813102e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .5286517579627517;
    b = .3075627775211328;
    v = 2.26243951663262e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .5586001195731895;
    b = .3382311089826877;
    v = 2.277874557231869e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .5872229902021319;
    b = .3681108834741399;
    v = 2.287854314454994e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .6144258616235123;
    b = .3970397446872839;
    v = 2.293268499615575e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .2951676508064861;
    b = .02867499538750441;
    v = 1.912628201529828e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .3335085485472725;
    b = .0586787934190351;
    v = 1.992499672238701e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .3709561760636381;
    b = .08961099205022284;
    v = 2.061275533454027e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .4074722861667498;
    b = .1211627927626297;
    v = 2.119318215968572e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .4429923648839117;
    b = .1530748903554898;
    v = 2.167416581882652e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .4774428052721736;
    b = .1851176436721877;
    v = 2.2064307305166e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .5107446539535904;
    b = .2170829107658179;
    v = 2.237186938699523e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .5428151370542935;
    b = .2487786689026271;
    v = 2.260480075032884e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .5735699292556964;
    b = .2800239952795016;
    v = 2.277098884558542e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .6029253794562866;
    b = .3106445702878119;
    v = 2.287845715109671e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .6307998987073145;
    b = .3404689500841194;
    v = 2.293547268236294e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .3752652273692719;
    b = .02997145098184479;
    v = 2.056073839852528e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .4135383879344028;
    b = .06086725898678011;
    v = 2.114235865831876e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .4506113885153907;
    b = .09238849548435643;
    v = 2.163175629770551e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .4864401554606072;
    b = .1242786603851851;
    v = 2.20339215811165e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .5209708076611709;
    b = .1563086731483386;
    v = 2.235473176847839e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .5541422135830122;
    b = .1882696509388506;
    v = 2.260024141501235e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .5858880915113817;
    b = .2199672979126059;
    v = 2.277675929329182e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .6161399390603444;
    b = .2512165482924867;
    v = 2.289102112284834e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .644829648225509;
    b = .2818368701871888;
    v = 2.295027954625118e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .4544796274917948;
    b = .03088970405060312;
    v = 2.161281589879992e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .4919389072146628;
    b = .06240947677636835;
    v = 2.201980477395102e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .5279313026985183;
    b = .09430706144280313;
    v = 2.234952066593166e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .5624169925571135;
    b = .1263547818770374;
    v = 2.260540098520838e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .5953484627093287;
    b = .1583430788822594;
    v = 2.279157981899988e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .6266730715339185;
    b = .1900748462555988;
    v = 2.291296918565571e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .6563363204278871;
    b = .2213599519592567;
    v = 2.297533752536649e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .5314574716585696;
    b = .03152508811515374;
    v = 2.234927356465995e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .5674614932298185;
    b = .06343865291465561;
    v = 2.261288012985219e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .6017706004970264;
    b = .09551503504223951;
    v = 2.280818160923688e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .6343471270264178;
    b = .1275440099801196;
    v = 2.293773295180159e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .6651494599127802;
    b = .159325203767196;
    v = 2.300528767338634e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .6050184986005704;
    b = .03192538338496105;
    v = 2.281893855065666e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .63901635508804;
    b = .06402824353962306;
    v = 2.295720444840727e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .6711199107088448;
    b = .09609805077002909;
    v = 2.303227649026753e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .6741354429572275;
    b = .03211853196273233;
    v = 2.304831913227114e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    --(*n);
    return 0;
}

int ld5294_(double *x, double *y, double *z, double *w, int *n)
{
    static double a, b, v;

    /* VW */
    /* VW    LEBEDEV 5294-POINT ANGULAR GRID */
    /* VW */
    /* hvd */
    /* hvd   This subroutine is part of a set of subroutines that generate */
    /* hvd   Lebedev grids [1-6] for integration on a sphere. The original */
    /* hvd   C-code [1] was kindly provided by Dr. Dmitri N. Laikov and */
    /* hvd   translated into fortran by Dr. Christoph van Wuellen. */
    /* hvd   This subroutine was translated using a C to fortran77 conversion */
    /* hvd   tool written by Dr. Christoph van Wuellen. */
    /* hvd */
    /* hvd   Users of this code are asked to include reference [1] in their */
    /* hvd   publications, and in the user- and programmers-manuals */
    /* hvd   describing their codes. */
    /* hvd */
    /* hvd   This code was distributed through CCL (http://www.ccl.net/). */
    /* hvd */
    /* hvd   [1] V.I. Lebedev, and D.N. Laikov */
    /* hvd       "A quadrature formula for the sphere of the 131st */
    /* hvd        algebraic order of accuracy" */
    /* hvd       Doklady Mathematics, Vol. 59, No. 3, 1999, pp. 477-481. */
    /* hvd */
    /* hvd   [2] V.I. Lebedev */
    /* hvd       "A quadrature formula for the sphere of 59th algebraic */
    /* hvd        order of accuracy" */
    /* hvd       Russian Acad. Sci. Dokl. Math., Vol. 50, 1995, pp. 283-286. */
    /* hvd */
    /* hvd   [3] V.I. Lebedev, and A.L. Skorokhodov */
    /* hvd       "Quadrature formulas of orders 41, 47, and 53 for the sphere" */
    /* hvd       Russian Acad. Sci. Dokl. Math., Vol. 45, 1992, pp. 587-592. */
    /* hvd */
    /* hvd   [4] V.I. Lebedev */
    /* hvd       "Spherical quadrature formulas exact to orders 25-29" */
    /* hvd       Siberian Mathematical Journal, Vol. 18, 1977, pp. 99-107. */
    /* hvd */
    /* hvd   [5] V.I. Lebedev */
    /* hvd       "Quadratures on a sphere" */
    /* hvd       Computational Mathematics and Mathematical Physics, Vol. 16, */
    /* hvd       1976, pp. 10-24. */
    /* hvd */
    /* hvd   [6] V.I. Lebedev */
    /* hvd       "Values of the nodes and weights of ninth to seventeenth */
    /* hvd        order Gauss-Markov quadrature formulae invariant under the */
    /* hvd        octahedron group with inversion" */
    /* hvd       Computational Mathematics and Mathematical Physics, Vol. 15, */
    /* hvd       1975, pp. 44-51. */
    /* hvd */
    /* Parameter adjustments */
    --w;
    --z;
    --y;
    --x;

    /* Function Body */
    *n = 1;
    v = 9.080510764308163e-5;
    gen_oh_(1, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    v = 2.084824361987793e-4;
    gen_oh_(3, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .0230326168626145;
    v = 5.011105657239616e-5;
    gen_oh_(4, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .03757208620162394;
    v = 5.942520409683854e-5;
    gen_oh_(4, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .05821912033821852;
    v = 9.564394826109721e-5;
    gen_oh_(4, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .08403127529194872;
    v = 1.185530657126338e-4;
    gen_oh_(4, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .1122927798060578;
    v = 1.364510114230331e-4;
    gen_oh_(4, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .1420125319192987;
    v = 1.505828825605415e-4;
    gen_oh_(4, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .1726396437341978;
    v = 1.619298749867023e-4;
    gen_oh_(4, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .2038170058115696;
    v = 1.712450504267789e-4;
    gen_oh_(4, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .2352849892876508;
    v = 1.789891098164999e-4;
    gen_oh_(4, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .2668363354312461;
    v = 1.854474955629795e-4;
    gen_oh_(4, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .2982941279900452;
    v = 1.908148636673661e-4;
    gen_oh_(4, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .3295002922087076;
    v = 1.952377405281833e-4;
    gen_oh_(4, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .3603094918363593;
    v = 1.988349254282232e-4;
    gen_oh_(4, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .390585789517392;
    v = 2.01707980716005e-4;
    gen_oh_(4, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .4202005758160837;
    v = 2.039473082709094e-4;
    gen_oh_(4, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .4490310061597227;
    v = 2.056360279288953e-4;
    gen_oh_(4, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .4769586160311491;
    v = 2.068525823066865e-4;
    gen_oh_(4, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .503867988704975;
    v = 2.076724877534488e-4;
    gen_oh_(4, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .5296454286519961;
    v = 2.081694278237885e-4;
    gen_oh_(4, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .554177620716485;
    v = 2.084157631219326e-4;
    gen_oh_(4, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .5990467321921213;
    v = 2.084381531128593e-4;
    gen_oh_(4, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .6191467096294587;
    v = 2.083476277129307e-4;
    gen_oh_(4, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .6375251212901849;
    v = 2.082686194459732e-4;
    gen_oh_(4, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .6540514381131168;
    v = 2.082475686112415e-4;
    gen_oh_(4, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .668589906439151;
    v = 2.083139860289915e-4;
    gen_oh_(4, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .6810013009681648;
    v = 2.084745561831237e-4;
    gen_oh_(4, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .691146957873034;
    v = 2.08709131337589e-4;
    gen_oh_(4, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .6988956915141736;
    v = 2.089718413297697e-4;
    gen_oh_(4, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .704133579486872;
    v = 2.092003303479793e-4;
    gen_oh_(4, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .7067754398018567;
    v = 2.093336148263241e-4;
    gen_oh_(4, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .03840368707853623;
    v = 7.591708117365267e-5;
    gen_oh_(5, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .09835485954117399;
    v = 1.083383968169186e-4;
    gen_oh_(5, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .1665774947612998;
    v = 1.40301939529251e-4;
    gen_oh_(5, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .240570233536291;
    v = 1.615970179286436e-4;
    gen_oh_(5, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .3165270770189046;
    v = 1.771144187504911e-4;
    gen_oh_(5, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .3927386145645443;
    v = 1.887760022988168e-4;
    gen_oh_(5, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .4678825918374656;
    v = 1.973474670768214e-4;
    gen_oh_(5, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .5408022024266935;
    v = 2.033787661234659e-4;
    gen_oh_(5, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .6104967445752438;
    v = 2.072343626517331e-4;
    gen_oh_(5, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .6760910702685738;
    v = 2.091177834226918e-4;
    gen_oh_(5, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .06655644120217392;
    b = .01936508874588424;
    v = 9.316684484675566e-5;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .09446246161270182;
    b = .04252442002115869;
    v = 1.116193688682976e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .1242651925452509;
    b = .06806529315354374;
    v = 1.298623551559414e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .1553438064846751;
    b = .09560957491205369;
    v = 1.450236832456426e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .187113711054267;
    b = .1245931657452888;
    v = 1.572719958149914e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .2192612628836257;
    b = .1545385828778978;
    v = 1.673234785867195e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .2515682807206955;
    b = .1851004249723368;
    v = 1.756860118725188e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .283853586628729;
    b = .2160182608272384;
    v = 1.826776290439367e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .3159578817528521;
    b = .2470799012277111;
    v = 1.885116347992865e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .3477370882791392;
    b = .2781014208986402;
    v = 1.933457860170574e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .379057696089054;
    b = .3089172523515731;
    v = 1.973060671902064e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .40979383178102;
    b = .3393750055472244;
    v = 2.004987099616311e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .4398256572859637;
    b = .369332247098773;
    v = 2.030170909281499e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .469038411471848;
    b = .3986541005609877;
    v = 2.04946146011908e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .4973216048301053;
    b = .4272112491408562;
    v = 2.063653565200186e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .5245681526132446;
    b = .4548781735309936;
    v = 2.073507927381027e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .5506733911803888;
    b = .4815315355023251;
    v = 2.079764593256122e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .5755339829522475;
    b = .5070486445801855;
    v = 2.083150534968778e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .1305472386056362;
    b = .02284970375722366;
    v = 1.262715121590664e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .1637327908216477;
    b = .04812254338288384;
    v = 1.414386128545972e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .1972734634149637;
    b = .07531734457511935;
    v = 1.538740401313898e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .230869465311013;
    b = .1039043639882017;
    v = 1.642434942331432e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .264389921833816;
    b = .1334526587117626;
    v = 1.729790609237496e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .2977171599622171;
    b = .1636414868936382;
    v = 1.803505190260828e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .330729390303231;
    b = .1942195406166568;
    v = 1.865475350079657e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .3633069198219073;
    b = .2249752879943753;
    v = 1.917182669679069e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .3953346955922727;
    b = .2557218821820032;
    v = 1.959851709034382e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .4267018394184914;
    b = .2862897925213193;
    v = 1.994529548117882e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .4573009622571704;
    b = .3165224536636518;
    v = 2.022138911146548e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .4870279559856109;
    b = .3462730221636496;
    v = 2.043518024208592e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .5157819581450322;
    b = .3754016870282835;
    v = 2.05945031301811e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .5434651666465393;
    b = .4037733784993613;
    v = 2.070685715318472e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .5699823887764627;
    b = .4312557784139123;
    v = 2.077955310694373e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .5952403350947741;
    b = .457717536712211;
    v = 2.081980387824712e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .2025152599210369;
    b = .02520253617719557;
    v = 1.521318610377956e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .2381066653274425;
    b = .05223254506119;
    v = 1.622772720185755e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .2732823383651612;
    b = .0806066968858862;
    v = 1.710498139420709e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .3080137692611118;
    b = .1099335754081255;
    v = 1.785911149448736e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .3422405614587601;
    b = .1399120955959857;
    v = 1.850125313687736e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .375880877389042;
    b = .1702977801651705;
    v = 1.904229703933298e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .4088458383438932;
    b = .200879925660168;
    v = 1.949259956121987e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .4410450550841152;
    b = .2314703052180836;
    v = 1.98616154536396e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .4723879420561312;
    b = .2618972111375892;
    v = 2.01579058564137e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .5027843561874343;
    b = .292001319560027;
    v = 2.038934198707418e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .5321453674452458;
    b = .3216322555190551;
    v = 2.056334060538251e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .560383911383403;
    b = .3506456615934198;
    v = 2.068705959462289e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .5874150706875146;
    b = .3789007181306267;
    v = 2.076753906106002e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .6131559381660038;
    b = .4062580170572782;
    v = 2.081179391734803e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .2778497016394506;
    b = .02696271276876226;
    v = 1.700345216228943e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .3143733562261912;
    b = .05523469316960465;
    v = 1.77490677999041e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .3501485810261827;
    b = .08445193201626464;
    v = 1.839659377002642e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .3851430322303653;
    b = .1143263119336083;
    v = 1.894987462975169e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .4193013979470415;
    b = .1446177898344475;
    v = 1.941548809452595e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .4525585960458567;
    b = .1751165438438091;
    v = 1.980078427252384e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .4848447779622947;
    b = .205633830674566;
    v = 2.011296284744488e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .5160871208276894;
    b = .2359965487229226;
    v = 2.035888456966776e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .5462112185696926;
    b = .2660430223139146;
    v = 2.054516325352142e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .5751425068101757;
    b = .2956193664498032;
    v = 2.067831033092635e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .6028073872853596;
    b = .3245763905312779;
    v = 2.076485320284876e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .6291338275278409;
    b = .3527670026206972;
    v = 2.081141439525255e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .3541797528439391;
    b = .0282385347943555;
    v = 1.834383015469222e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .3908234972074657;
    b = .05741296374713106;
    v = 1.889540591777677e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .426440845010759;
    b = .08724646633650199;
    v = 1.936677023597375e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .4609949666553286;
    b = .1175034422915616;
    v = 1.976176495066504e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .4944389496536006;
    b = .1479755652628428;
    v = 2.008536004560983e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .5267194884346086;
    b = .1784740659484352;
    v = 2.034280351712291e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .557778781022099;
    b = .2088245700431244;
    v = 2.053944466027758e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .587556376353667;
    b = .2388628136570763;
    v = 2.06807764288236e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .6159910016391269;
    b = .2684308928769185;
    v = 2.077250949661599e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .6430219602956268;
    b = .2973740761960252;
    v = 2.08206244070532e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .4300647036213646;
    b = .02916399920493977;
    v = 1.934374486546626e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .4661486308935531;
    b = .05898803024755659;
    v = 1.9741070104843e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .5009658555287261;
    b = .08924162698525409;
    v = 2.007129290388658e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .5344824270447704;
    b = .1197185199637321;
    v = 2.033736947471293e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .5666575997416371;
    b = .1502300756161382;
    v = 2.054287125902493e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .5974457471404752;
    b = .1806004191913564;
    v = 2.069184936818894e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .6267984444116886;
    b = .2106621764786252;
    v = 2.078883689808782e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .6546664713575417;
    b = .2402526932671914;
    v = 2.083886366116359e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .5042711004437253;
    b = .02982529203607657;
    v = 2.006593275470817e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .539212745677438;
    b = .06008728062339922;
    v = 2.033728426135397e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .5726819437668618;
    b = .09058227674571398;
    v = 2.055008781377608e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .6046469254207278;
    b = .12112192358034;
    v = 2.070651783518502e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .6350716157434952;
    b = .151528640479158;
    v = 2.08095333509432e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .6639177679185454;
    b = .1816314681255552;
    v = 2.086284998988521e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .5757276040972253;
    b = .0302699175257544;
    v = 2.055549387644668e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .6090265823139755;
    b = .0607840229787077;
    v = 2.071871850267654e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .6406735344387661;
    b = .09135459984176636;
    v = 2.082856600431965e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .6706397927793709;
    b = .121802415596659;
    v = 2.088705858819358e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .6435019674426665;
    b = .03052608357660639;
    v = 2.083995867536322e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .6747218676375681;
    b = .06112185773983089;
    v = 2.090509712889637e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    --(*n);
    return 0;
}

int ld5810_(double *x, double *y, double *z, double *w, int *n)
{
    static double a, b, v;

    /* VW */
    /* VW    LEBEDEV 5810-POINT ANGULAR GRID */
    /* VW */
    /* hvd */
    /* hvd   This subroutine is part of a set of subroutines that generate */
    /* hvd   Lebedev grids [1-6] for integration on a sphere. The original */
    /* hvd   C-code [1] was kindly provided by Dr. Dmitri N. Laikov and */
    /* hvd   translated into fortran by Dr. Christoph van Wuellen. */
    /* hvd   This subroutine was translated using a C to fortran77 conversion */
    /* hvd   tool written by Dr. Christoph van Wuellen. */
    /* hvd */
    /* hvd   Users of this code are asked to include reference [1] in their */
    /* hvd   publications, and in the user- and programmers-manuals */
    /* hvd   describing their codes. */
    /* hvd */
    /* hvd   This code was distributed through CCL (http://www.ccl.net/). */
    /* hvd */
    /* hvd   [1] V.I. Lebedev, and D.N. Laikov */
    /* hvd       "A quadrature formula for the sphere of the 131st */
    /* hvd        algebraic order of accuracy" */
    /* hvd       Doklady Mathematics, Vol. 59, No. 3, 1999, pp. 477-481. */
    /* hvd */
    /* hvd   [2] V.I. Lebedev */
    /* hvd       "A quadrature formula for the sphere of 59th algebraic */
    /* hvd        order of accuracy" */
    /* hvd       Russian Acad. Sci. Dokl. Math., Vol. 50, 1995, pp. 283-286. */
    /* hvd */
    /* hvd   [3] V.I. Lebedev, and A.L. Skorokhodov */
    /* hvd       "Quadrature formulas of orders 41, 47, and 53 for the sphere" */
    /* hvd       Russian Acad. Sci. Dokl. Math., Vol. 45, 1992, pp. 587-592. */
    /* hvd */
    /* hvd   [4] V.I. Lebedev */
    /* hvd       "Spherical quadrature formulas exact to orders 25-29" */
    /* hvd       Siberian Mathematical Journal, Vol. 18, 1977, pp. 99-107. */
    /* hvd */
    /* hvd   [5] V.I. Lebedev */
    /* hvd       "Quadratures on a sphere" */
    /* hvd       Computational Mathematics and Mathematical Physics, Vol. 16, */
    /* hvd       1976, pp. 10-24. */
    /* hvd */
    /* hvd   [6] V.I. Lebedev */
    /* hvd       "Values of the nodes and weights of ninth to seventeenth */
    /* hvd        order Gauss-Markov quadrature formulae invariant under the */
    /* hvd        octahedron group with inversion" */
    /* hvd       Computational Mathematics and Mathematical Physics, Vol. 15, */
    /* hvd       1975, pp. 44-51. */
    /* hvd */
    /* Parameter adjustments */
    --w;
    --z;
    --y;
    --x;

    /* Function Body */
    *n = 1;
    v = 9.735347946175486e-6;
    gen_oh_(1, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    v = 1.907581241803167e-4;
    gen_oh_(2, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    v = 1.901059546737578e-4;
    gen_oh_(3, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .01182361662400277;
    v = 3.926424538919212e-5;
    gen_oh_(4, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .03062145009138958;
    v = 6.667905467294382e-5;
    gen_oh_(4, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .05329794036834243;
    v = 8.868891315019135e-5;
    gen_oh_(4, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .0784816553286222;
    v = 1.066306000958872e-4;
    gen_oh_(4, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .1054038157636201;
    v = 1.214506743336128e-4;
    gen_oh_(4, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .1335577797766211;
    v = 1.338054681640871e-4;
    gen_oh_(4, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .1625769955502252;
    v = 1.441677023628504e-4;
    gen_oh_(4, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .1921787193412792;
    v = 1.528880200826557e-4;
    gen_oh_(4, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .2221340534690548;
    v = 1.602330623773609e-4;
    gen_oh_(4, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .2522504912791132;
    v = 1.664102653445244e-4;
    gen_oh_(4, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .2823610860679697;
    v = 1.715845854011323e-4;
    gen_oh_(4, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .312317396626756;
    v = 1.758901000133069e-4;
    gen_oh_(4, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .3419847036953789;
    v = 1.794382485256736e-4;
    gen_oh_(4, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .3712386456999758;
    v = 1.823238106757407e-4;
    gen_oh_(4, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .3999627649876828;
    v = 1.846293252959976e-4;
    gen_oh_(4, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .4280466458648093;
    v = 1.864284079323098e-4;
    gen_oh_(4, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .4553844360185711;
    v = 1.877882694626914e-4;
    gen_oh_(4, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .4818736094437834;
    v = 1.887716321852025e-4;
    gen_oh_(4, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .5074138709260629;
    v = 1.894381638175673e-4;
    gen_oh_(4, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .5319061304570707;
    v = 1.898454899533629e-4;
    gen_oh_(4, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .5552514978677286;
    v = 1.900497929577815e-4;
    gen_oh_(4, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .5981009025246183;
    v = 1.900671501924092e-4;
    gen_oh_(4, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .6173990192228116;
    v = 1.89983755553351e-4;
    gen_oh_(4, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .6351365239411131;
    v = 1.899014113156229e-4;
    gen_oh_(4, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .65120102282272;
    v = 1.898581257705106e-4;
    gen_oh_(4, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .665475836394812;
    v = 1.898804756095753e-4;
    gen_oh_(4, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .677841041485337;
    v = 1.899793610426402e-4;
    gen_oh_(4, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .688176088748411;
    v = 1.901464554844117e-4;
    gen_oh_(4, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .6963645267094598;
    v = 1.903533246259542e-4;
    gen_oh_(4, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .7023010617153579;
    v = 1.905556158463228e-4;
    gen_oh_(4, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .7059004636628753;
    v = 1.907037155663528e-4;
    gen_oh_(4, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .03552470312472575;
    v = 5.992997844249967e-5;
    gen_oh_(5, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .09151176620841283;
    v = 9.749059382456978e-5;
    gen_oh_(5, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .156619793006898;
    v = 1.241680804599158e-4;
    gen_oh_(5, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .2265467599271907;
    v = 1.43762615429936e-4;
    gen_oh_(5, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .2988242318581361;
    v = 1.584200054793902e-4;
    gen_oh_(5, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .3717482419703886;
    v = 1.694436550982744e-4;
    gen_oh_(5, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .4440094491758889;
    v = 1.776617014018108e-4;
    gen_oh_(5, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .5145337096756642;
    v = 1.836132434440077e-4;
    gen_oh_(5, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .582405367286023;
    v = 1.876494727075983e-4;
    gen_oh_(5, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .646828396104337;
    v = 1.899906535336482e-4;
    gen_oh_(5, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .06095964259104373;
    b = .01787828275342931;
    v = 8.14325282076735e-5;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .08811962270959388;
    b = .03953888740792096;
    v = 9.998859890887728e-5;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .1165936722428831;
    b = .0637812179772299;
    v = 1.156199403068359e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .1460232857031785;
    b = .08985890813745037;
    v = 1.287632092635513e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .1761197110181755;
    b = .1172606510576162;
    v = 1.398378643365139e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .2066471190463718;
    b = .1456102876970995;
    v = 1.491876468417391e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .2374076026328152;
    b = .1746153823011775;
    v = 1.570855679175456e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .2682305474337051;
    b = .2040383070295584;
    v = 1.637483948103775e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .2989653312142369;
    b = .2336788634003698;
    v = 1.693500566632843e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .3294762752772209;
    b = .2633632752654219;
    v = 1.740322769393633e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .3596390887276086;
    b = .2929369098051601;
    v = 1.779126637278296e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .3893383046398812;
    b = .3222592785275512;
    v = 1.810908108835412e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .4184653789358347;
    b = .3512004791195743;
    v = 1.83652913260019e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .4469172319076166;
    b = .3796385677684537;
    v = 1.856752841777379e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .4745950813276976;
    b = .4074575378263879;
    v = 1.872270566606832e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .5014034601410262;
    b = .4345456906027828;
    v = 1.883722645591307e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .5272493404551239;
    b = .4607942515205134;
    v = 1.891714324525297e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .5520413051846366;
    b = .486096128418172;
    v = 1.896827480450146e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .5756887237503077;
    b = .510344739534279;
    v = 1.899628417059528e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .1225039430588352;
    b = .02136455922655793;
    v = 1.123301829001669e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .1539113217321372;
    b = .04520926166137188;
    v = 1.253698826711277e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .1856213098637712;
    b = .07086468177864818;
    v = 1.366266117678531e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .2174998728035131;
    b = .09785239488772918;
    v = 1.462736856106918e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .249412833693833;
    b = .125810639626721;
    v = 1.545076466685412e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .281232156214348;
    b = .1544529125047001;
    v = 1.615096280814007e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .3128372276456111;
    b = .1835433512202753;
    v = 1.674366639741759e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .3441145160177973;
    b = .2128813258619585;
    v = 1.7242250024379e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .374956771485351;
    b = .2422913734880829;
    v = 1.765810822987288e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .405262173201561;
    b = .2716163748391453;
    v = 1.800104126010751e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .4349335453522385;
    b = .300712767124028;
    v = 1.827960437331284e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .4638776641524965;
    b = .3294470677216479;
    v = 1.850140300716308e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .4920046410462687;
    b = .3576932543699155;
    v = 1.867333507394938e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .5192273554861704;
    b = .3853307059757764;
    v = 1.880178688638289e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .5454609081136522;
    b = .4122425044452694;
    v = 1.889278925654758e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .570622066142414;
    b = .4383139587781027;
    v = 1.895213832507346e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .5946286755181518;
    b = .4634312536300553;
    v = 1.89854827739742e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .1905370790924295;
    b = .02371311537781979;
    v = 1.349105935937341e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .2242518717748009;
    b = .04917878059254806;
    v = 1.444060068369326e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .2577190808025936;
    b = .07595498960495142;
    v = 1.526797390930008e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .2908724534927187;
    b = .10369910831911;
    v = 1.598208771406474e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .3236354020056219;
    b = .1321348584450234;
    v = 1.659354368615331e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .3559267359304543;
    b = .1610316571314789;
    v = 1.71127991094644e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .3876637123676956;
    b = .1901912080395707;
    v = 1.75495272560144e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .4187636705218842;
    b = .219438495013795;
    v = 1.791247850802529e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .4491449019883107;
    b = .2486155334763858;
    v = 1.820954300877716e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .4787270932425445;
    b = .2775768931812335;
    v = 1.844788524548449e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .5074315153055574;
    b = .306186378659112;
    v = 1.86340948170622e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .5351810507738336;
    b = .3343144718152556;
    v = 1.877433008795068e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .5619001025975381;
    b = .3618362729028427;
    v = 1.887444543705232e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .5875144035268046;
    b = .3886297583620408;
    v = 1.894009829375006e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .6119507308734495;
    b = .4145742277792031;
    v = 1.897683345035198e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .2619733870119463;
    b = .02540047186389353;
    v = 1.517327037467653e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .2968149743237949;
    b = .05208107018543989;
    v = 1.587740557483543e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .3310451504860488;
    b = .07971828470885599;
    v = 1.649093382274097e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .3646215567376676;
    b = .1080465999177927;
    v = 1.701915216193265e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .397491678527936;
    b = .1368413849366629;
    v = 1.746847753144065e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .4295967403772029;
    b = .1659073184763559;
    v = 1.78455551200757e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .4608742854473447;
    b = .1950703730454614;
    v = 1.815687562112174e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .4912598858949903;
    b = .2241721144376724;
    v = 1.840864370663302e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .5206882758945558;
    b = .2530655255406489;
    v = 1.860676785390006e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .5490940914019819;
    b = .2816118409731066;
    v = 1.875690583743703e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .5764123302025542;
    b = .3096780504593238;
    v = 1.886453236347225e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .6025786004213506;
    b = .3371348366394987;
    v = 1.893501123329645e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .6275291964794956;
    b = .3638547827694396;
    v = 1.897366184519868e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .3348189479861771;
    b = .02664841935537443;
    v = 1.643908815152736e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .3699515545855295;
    b = .05424000066843495;
    v = 1.696300350907768e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .4042003071474669;
    b = .08251992715430854;
    v = 1.741553103844483e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .4375320100182624;
    b = .111269518248371;
    v = 1.780015282386092e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .4699054490335947;
    b = .1402964116467816;
    v = 1.812116787077125e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .5012739879431952;
    b = .1694275117584291;
    v = 1.838323158085421e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .5315874883754966;
    b = .1985038235312689;
    v = 1.859113119837737e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .5607937109622117;
    b = .2273765660020893;
    v = 1.874969220221698e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .5888393223495521;
    b = .2559041492849764;
    v = 1.886375612681076e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .6156705979160163;
    b = .2839497251976899;
    v = 1.893819575809276e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .6412338809078123;
    b = .311379106050069;
    v = 1.897794748256767e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .4076051259257167;
    b = .02757792290858463;
    v = 1.738963926584846e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .442378812579152;
    b = .05584136834984293;
    v = 1.777442359873466e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .4760480917328258;
    b = .08457772087727143;
    v = 1.810010815068719e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .5085838725946297;
    b = .1135975846359248;
    v = 1.836920318248129e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .5399513637391218;
    b = .1427286904765053;
    v = 1.858489473214328e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .570111843363638;
    b = .1718112740057635;
    v = 1.875079342496592e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .5990240530606021;
    b = .2006944855985351;
    v = 1.88708023910231e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .6266452685139695;
    b = .2292335090598907;
    v = 1.894905752176822e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .6529320971415942;
    b = .2572871512353714;
    v = 1.898991061200695e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .4791583834610126;
    b = .02826094197735932;
    v = 1.809065016458791e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .513037395279694;
    b = .05699871359683649;
    v = 1.836297121596799e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .5456252429628476;
    b = .08602712528554394;
    v = 1.858426916241869e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .5768956329682385;
    b = .1151748137221281;
    v = 1.875654101134641e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .6068186944699046;
    b = .1442811654136362;
    v = 1.888240751833503e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .6353622248024907;
    b = .173193032165768;
    v = 1.896497383866979e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .6624927035731797;
    b = .2017619958756061;
    v = 1.900775530219121e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .5484933508028488;
    b = .02874219755907391;
    v = 1.858525041478814e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .5810207682142106;
    b = .05778312123713695;
    v = 1.876248690077947e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .6120955197181352;
    b = .08695262371439526;
    v = 1.889404439064607e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .6416944284294319;
    b = .1160893767057166;
    v = 1.89816853926529e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .669792639173126;
    b = .1450378826743251;
    v = 1.902779940661772e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .6147594390585488;
    b = .02904957622341456;
    v = 1.890125641731815e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .6455390026356783;
    b = .05823809152617197;
    v = 1.899434637795751e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .6747258588365477;
    b = .08740384899884715;
    v = 1.904520856831751e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    a = .6772135750395347;
    b = .02919946135808105;
    v = 1.905534498734563e-4;
    gen_oh_(6, n, &x[*n], &y[*n], &z[*n], &w[*n], &a, &b, &v);
    --(*n);
    return 0;
}
