#ifndef lint
static char vcid[] = "$Id: aijfact.c,v 1.31 1995/08/23 17:14:15 curfman Exp curfman $";
#endif


#include "aij.h"
#include "inline/spops.h"
/*
    Factorization code for AIJ format. 
*/

int MatLUFactorSymbolic_AIJ(Mat mat,IS isrow,IS iscol,double f,Mat *fact)
{
  Mat_AIJ *aij = (Mat_AIJ *) mat->data, *aijnew;
  IS      isicol;
  int     *r,*ic, ierr, i, n = aij->m, *ai = aij->i, *aj = aij->j;
  int     *ainew,*ajnew, jmax,*fill, *ajtmp, nz;
  int     *idnew, idx, row,m,fm, nnz, nzi,len, realloc = 0,nzbd,*im;
 
  if (n != aij->n) 
    SETERRQ(1,"MatLUFactorSymbolic_AIJ:Matrix must be square");
  if (!isrow) 
    SETERRQ(1,"MatLUFactorSymbolic_AIJ:Matrix must have row permutation");
  if (!iscol) 
    SETERRQ(1,"MatLUFactorSymbolic_AIJ:Matrix must have column permutation");

  ierr = ISInvertPermutation(iscol,&isicol); CHKERRQ(ierr);
  ISGetIndices(isrow,&r); ISGetIndices(isicol,&ic);

  /* get new row pointers */
  ainew = (int *) PETSCMALLOC( (n+1)*sizeof(int) ); CHKPTRQ(ainew);
  ainew[0] = 1;
  /* don't know how many column pointers are needed so estimate */
  jmax = (int) (f*ai[n]);
  ajnew = (int *) PETSCMALLOC( (jmax)*sizeof(int) ); CHKPTRQ(ajnew);
  /* fill is a linked list of nonzeros in active row */
  fill = (int *) PETSCMALLOC( (2*n+1)*sizeof(int)); CHKPTRQ(fill);
  im = fill + n + 1;
  /* idnew is location of diagonal in factor */
  idnew = (int *) PETSCMALLOC( (n+1)*sizeof(int)); CHKPTRQ(idnew);
  idnew[0] = 1;

  for ( i=0; i<n; i++ ) {
    /* first copy previous fill into linked list */
    nnz = nz    = ai[r[i]+1] - ai[r[i]];
    ajtmp = aj + ai[r[i]] - 1;
    fill[n] = n;
    while (nz--) {
      fm = n;
      idx = ic[*ajtmp++ - 1];
      do {
        m = fm;
        fm = fill[m];
      } while (fm < idx);
      fill[m] = idx;
      fill[idx] = fm;
    }
    row = fill[n];
    while ( row < i ) {
      ajtmp = ajnew + idnew[row];
      nzbd = 1 + idnew[row] - ainew[row];
      nz = im[row] - nzbd;
      fm = row;
      while (nz-- > 0) {
        /* fm = n;  */
        idx = *ajtmp++ - 1;
        nzbd++;
        if (idx == i) im[row] = nzbd;
        do {
          m = fm;
          fm = fill[m];
        } while (fm < idx);
        if (fm != idx) {
          fill[m] = idx;
          fill[idx] = fm;
          fm = idx;
          nnz++;
        }
/*  printf("i %d row %d nz %d idx %d fm %d\n",i,row,nz,idx,fm);  */
      }
      row = fill[row];
    }
    /* copy new filled row into permanent storage */
    ainew[i+1] = ainew[i] + nnz;
    if (ainew[i+1] > jmax+1) {
      /* allocate a longer ajnew */
      int maxadd;
      maxadd = (int) ((f*ai[n]*(n-i+5))/n);
      if (maxadd < nnz) maxadd = (n-i)*(nnz+1);
      jmax += maxadd;
      ajtmp = (int *) PETSCMALLOC( jmax*sizeof(int) );CHKPTRQ(ajtmp);
      PETSCMEMCPY(ajtmp,ajnew,(ainew[i]-1)*sizeof(int));
      PETSCFREE(ajnew);
      ajnew = ajtmp;
      realloc++; /* count how many times we realloc */
    }
    ajtmp = ajnew + ainew[i] - 1;
    fm = fill[n];
    nzi = 0;
    im[i] = nnz;
    while (nnz--) {
      if (fm < i) nzi++;
      *ajtmp++ = fm + 1;
      fm = fill[fm];
    }
    idnew[i] = ainew[i] + nzi;
  }

  PLogInfo((PetscObject)mat,
    "Info:MatLUFactorSymbolic_AIJ:Reallocs %d Fill ratio:given %g needed %g\n",
                             realloc,f,((double)ainew[n])/((double)ai[i]));

  ierr = ISRestoreIndices(isrow,&r); CHKERRQ(ierr);
  ierr = ISRestoreIndices(isicol,&ic); CHKERRQ(ierr);
  ierr = ISDestroy(isicol); CHKERRQ(ierr);
  PETSCFREE(fill);

  /* put together the new matrix */
  ierr = MatCreateSequentialAIJ(mat->comm,n, n, 0, 0, fact); CHKERRQ(ierr);
  aijnew = (Mat_AIJ *) (*fact)->data;
  PETSCFREE(aijnew->imax);
  aijnew->singlemalloc = 0;
  len = (ainew[n] - 1)*sizeof(Scalar);
  /* the next line frees the default space generated by the Create() */
  PETSCFREE(aijnew->a); PETSCFREE(aijnew->ilen);
  aijnew->a          = (Scalar *) PETSCMALLOC( len ); CHKPTRQ(aijnew->a);
  aijnew->j          = ajnew;
  aijnew->i          = ainew;
  aijnew->diag       = idnew;
  aijnew->ilen       = 0;
  aijnew->imax       = 0;
  aijnew->row        = isrow;
  aijnew->col        = iscol;
  aijnew->solve_work = (Scalar *) PETSCMALLOC( n*sizeof(Scalar)); 
  CHKPTRQ(aijnew->solve_work);
  /* In aijnew structure:  Free imax, ilen, old a, old j.  
     Allocate idnew, solve_work, new a, new j */
  PLogObjectMemory(*fact,(ainew[n]-1-n)*(sizeof(int)+sizeof(Scalar)));
  aijnew->maxnz = aijnew->nz = ainew[n] - 1;

  /* Cannot do this here because child is destroyed before parent created
     PLogObjectParent(*fact,isicol); */
  return 0; 
}

int MatLUFactorNumeric_AIJ(Mat mat,Mat *infact)
{
  Mat     fact = *infact;
  Mat_AIJ *aij = (Mat_AIJ *) mat->data, *aijnew = (Mat_AIJ *)fact->data;
  IS      iscol = aijnew->col, isrow = aijnew->row, isicol;
  int     *r,*ic, ierr, i, j, n = aij->m, *ai = aijnew->i, *aj = aijnew->j;
  int     *ajtmpold, *ajtmp, nz, row,*pj;
  Scalar  *rtmp,*v, *pv, *pc, multiplier; 

  ierr = ISInvertPermutation(iscol,&isicol); CHKERRQ(ierr);
  PLogObjectParent(*infact,isicol);
  ierr = ISGetIndices(isrow,&r); CHKERRQ(ierr);
  ierr = ISGetIndices(isicol,&ic); CHKERRQ(ierr);
  rtmp = (Scalar *) PETSCMALLOC( (n+1)*sizeof(Scalar) ); CHKPTRQ(rtmp);

  for ( i=0; i<n; i++ ) {
    nz = ai[i+1] - ai[i];
    ajtmp = aj + ai[i] - 1;
    for  ( j=0; j<nz; j++ ) rtmp[ajtmp[j]-1] = 0.0;

    /* load in initial (unfactored row) */
    nz = aij->i[r[i]+1] - aij->i[r[i]];
    ajtmpold = aij->j + aij->i[r[i]] - 1;
    v  = aij->a + aij->i[r[i]] - 1;
    for ( j=0; j<nz; j++ ) rtmp[ic[ajtmpold[j]-1]] =  v[j];

    row = *ajtmp++ - 1;
    while (row < i) {
      pc = rtmp + row;
      if (*pc != 0.0) {
        nz = aijnew->diag[row] - ai[row];
        pv = aijnew->a + aijnew->diag[row] - 1;
        pj = aijnew->j + aijnew->diag[row];
        multiplier = *pc * *pv++;
        *pc = multiplier;
        nz = ai[row+1] - ai[row] - 1 - nz;
        PLogFlops(2*nz);
        while (nz-->0) rtmp[*pj++ - 1] -= multiplier* *pv++; 
      }      
      row = *ajtmp++ - 1;
    }
    /* finished row so stick it into aijnew->a */
    pv = aijnew->a + ai[i] - 1;
    pj = aijnew->j + ai[i] - 1;
    nz = ai[i+1] - ai[i];
    if (rtmp[i] == 0.0) {SETERRQ(1,"MatLUFactorNumeric_AIJ:Zero pivot");}
    rtmp[i] = 1.0/rtmp[i];
    for ( j=0; j<nz; j++ ) {pv[j] = rtmp[pj[j]-1];}
  } 
  PETSCFREE(rtmp);
  ierr = ISRestoreIndices(isicol,&ic); CHKERRQ(ierr);
  ierr = ISRestoreIndices(isrow,&r); CHKERRQ(ierr);
  ierr = ISDestroy(isicol); CHKERRQ(ierr);
  fact->factor      = FACTOR_LU;
  aijnew->assembled = 1;
  PLogFlops(aijnew->n);
  return 0;
}
int MatLUFactor_AIJ(Mat matin,IS row,IS col,double f)
{
  Mat_AIJ *mat = (Mat_AIJ *) matin->data;
  int     ierr;
  Mat     fact;
  ierr = MatLUFactorSymbolic_AIJ(matin,row,col,f,&fact); CHKERRQ(ierr);
  ierr = MatLUFactorNumeric_AIJ(matin,&fact); CHKERRQ(ierr);

  /* free all the data structures from mat */
  PETSCFREE(mat->a); 
  if (!mat->singlemalloc) {PETSCFREE(mat->i); PETSCFREE(mat->j);}
  if (mat->diag) PETSCFREE(mat->diag);
  if (mat->ilen) PETSCFREE(mat->ilen);
  if (mat->imax) PETSCFREE(mat->imax);
  PETSCFREE(mat);

  PETSCMEMCPY(matin,fact,sizeof(struct _Mat));
  PETSCFREE(fact);
  return 0;
}

int MatSolve_AIJ(Mat mat,Vec bb, Vec xx)
{
  Mat_AIJ *aij = (Mat_AIJ *) mat->data;
  IS      iscol = aij->col, isrow = aij->row;
  int     *r,*c, ierr, i,  n = aij->m, *vi, *ai = aij->i, *aj = aij->j;
  int     nz;
  Scalar  *x,*b,*tmp, *aa = aij->a, sum, *v;

  if (mat->factor != FACTOR_LU) 
    SETERRQ(1,"MatSolve_AIJ:Cannot solve with unfactored matrix");

  ierr = VecGetArray(bb,&b); CHKERRQ(ierr);
  ierr = VecGetArray(xx,&x); CHKERRQ(ierr);
  tmp = aij->solve_work;

  ierr = ISGetIndices(isrow,&r);CHKERRQ(ierr);
  ierr = ISGetIndices(iscol,&c);CHKERRQ(ierr); c = c + (n-1);

  /* forward solve the lower triangular */
  tmp[0] = b[*r++];
  for ( i=1; i<n; i++ ) {
    v   = aa + ai[i] - 1;
    vi  = aj + ai[i] - 1;
    nz  = aij->diag[i] - ai[i];
    sum = b[*r++];
    while (nz--) sum -= *v++ * tmp[*vi++ - 1];
    tmp[i] = sum;
  }

  /* backward solve the upper triangular */
  for ( i=n-1; i>=0; i-- ){
    v   = aa + aij->diag[i];
    vi  = aj + aij->diag[i];
    nz  = ai[i+1] - aij->diag[i] - 1;
    sum = tmp[i];
    while (nz--) sum -= *v++ * tmp[*vi++ - 1];
    x[*c--] = tmp[i] = sum*aa[aij->diag[i]-1];
  }

  ierr = ISRestoreIndices(isrow,&r); CHKERRQ(ierr);
  ierr = ISRestoreIndices(iscol,&c); CHKERRQ(ierr);
  PLogFlops(2*aij->nz - aij->n);
  return 0;
}
int MatSolveAdd_AIJ(Mat mat,Vec bb, Vec yy, Vec xx)
{
  Mat_AIJ *aij = (Mat_AIJ *) mat->data;
  IS      iscol = aij->col, isrow = aij->row;
  int     *r,*c, ierr, i,  n = aij->m, *vi, *ai = aij->i, *aj = aij->j;
  int     nz;
  Scalar  *x,*b,*tmp, *aa = aij->a, sum, *v;

  if (mat->factor != FACTOR_LU) 
    SETERRQ(1,"MatSolveAdd_AIJ: Cannot solve with unfactored matrix");
  if (yy != xx) {ierr = VecCopy(yy,xx); CHKERRQ(ierr);}

  ierr = VecGetArray(bb,&b); CHKERRQ(ierr);
  ierr = VecGetArray(xx,&x); CHKERRQ(ierr);
  tmp = aij->solve_work;

  ierr = ISGetIndices(isrow,&r); CHKERRQ(ierr);
  ierr = ISGetIndices(iscol,&c); CHKERRQ(ierr); c = c + (n-1);

  /* forward solve the lower triangular */
  tmp[0] = b[*r++];
  for ( i=1; i<n; i++ ) {
    v   = aa + ai[i] - 1;
    vi  = aj + ai[i] - 1;
    nz  = aij->diag[i] - ai[i];
    sum = b[*r++];
    while (nz--) sum -= *v++ * tmp[*vi++ - 1];
    tmp[i] = sum;
  }

  /* backward solve the upper triangular */
  for ( i=n-1; i>=0; i-- ){
    v   = aa + aij->diag[i];
    vi  = aj + aij->diag[i];
    nz  = ai[i+1] - aij->diag[i] - 1;
    sum = tmp[i];
    while (nz--) sum -= *v++ * tmp[*vi++ - 1];
    tmp[i] = sum*aa[aij->diag[i]-1];
    x[*c--] += tmp[i];
  }

  ierr = ISRestoreIndices(isrow,&r); CHKERRQ(ierr);
  ierr = ISRestoreIndices(iscol,&c); CHKERRQ(ierr);
  PLogFlops(2*aij->nz);

  return 0;
}
/* -------------------------------------------------------------------*/
int MatSolveTrans_AIJ(Mat mat,Vec bb, Vec xx)
{
  Mat_AIJ *aij = (Mat_AIJ *) mat->data;
  IS      iscol = aij->col, isrow = aij->row, invisrow,inviscol;
  int     *r,*c, ierr, i, n = aij->m, *vi, *ai = aij->i, *aj = aij->j;
  int     nz;
  Scalar  *x,*b,*tmp, *aa = aij->a, *v;

  if (mat->factor != FACTOR_LU) 
    SETERRQ(1,"MatSolveTrans_AIJ:Cannot solve with unfactored matrix");
  ierr = VecGetArray(bb,&b); CHKERRQ(ierr);
  ierr = VecGetArray(xx,&x); CHKERRQ(ierr);
  tmp = aij->solve_work;

  /* invert the permutations */
  ierr = ISInvertPermutation(isrow,&invisrow); CHKERRQ(ierr);
  ierr = ISInvertPermutation(iscol,&inviscol); CHKERRQ(ierr);

  ierr = ISGetIndices(invisrow,&r); CHKERRQ(ierr);
  ierr = ISGetIndices(inviscol,&c); CHKERRQ(ierr);

  /* copy the b into temp work space according to permutation */
  for ( i=0; i<n; i++ ) tmp[c[i]] = b[i];

  /* forward solve the U^T */
  for ( i=0; i<n; i++ ) {
    v   = aa + aij->diag[i] - 1;
    vi  = aj + aij->diag[i];
    nz  = ai[i+1] - aij->diag[i] - 1;
    tmp[i] *= *v++;
    while (nz--) {
      tmp[*vi++ - 1] -= (*v++)*tmp[i];
    }
  }

  /* backward solve the L^T */
  for ( i=n-1; i>=0; i-- ){
    v   = aa + aij->diag[i] - 2;
    vi  = aj + aij->diag[i] - 2;
    nz  = aij->diag[i] - ai[i];
    while (nz--) {
      tmp[*vi-- - 1] -= (*v--)*tmp[i];
    }
  }

  /* copy tmp into x according to permutation */
  for ( i=0; i<n; i++ ) x[r[i]] = tmp[i];

  ierr = ISRestoreIndices(invisrow,&r); CHKERRQ(ierr);
  ierr = ISRestoreIndices(inviscol,&c); CHKERRQ(ierr);
  ierr = ISDestroy(invisrow); CHKERRQ(ierr);
  ierr = ISDestroy(inviscol); CHKERRQ(ierr);

  PLogFlops(2*aij->nz-aij->n);
  return 0;
}

int MatSolveTransAdd_AIJ(Mat mat,Vec bb, Vec zz,Vec xx)
{
  Mat_AIJ *aij = (Mat_AIJ *) mat->data;
  IS      iscol = aij->col, isrow = aij->row, invisrow,inviscol;
  int     *r,*c, ierr, i, n = aij->m, *vi, *ai = aij->i, *aj = aij->j;
  int     nz;
  Scalar  *x,*b,*tmp, *aa = aij->a, *v;

  if (mat->factor != FACTOR_LU) 
    SETERRQ(1,"MatSolveTransAdd_AIJ:Cannot solve with unfactored matrix");
  if (zz != xx) VecCopy(zz,xx);

  ierr = VecGetArray(bb,&b); CHKERRQ(ierr);
  ierr = VecGetArray(xx,&x); CHKERRQ(ierr);
  tmp = aij->solve_work;

  /* invert the permutations */
  ierr = ISInvertPermutation(isrow,&invisrow); CHKERRQ(ierr);
  ierr = ISInvertPermutation(iscol,&inviscol); CHKERRQ(ierr);
  ierr = ISGetIndices(invisrow,&r); CHKERRQ(ierr);
  ierr = ISGetIndices(inviscol,&c); CHKERRQ(ierr);

  /* copy the b into temp work space according to permutation */
  for ( i=0; i<n; i++ ) tmp[c[i]] = b[i];

  /* forward solve the U^T */
  for ( i=0; i<n; i++ ) {
    v   = aa + aij->diag[i] - 1;
    vi  = aj + aij->diag[i];
    nz  = ai[i+1] - aij->diag[i] - 1;
    tmp[i] *= *v++;
    while (nz--) {
      tmp[*vi++ - 1] -= (*v++)*tmp[i];
    }
  }

  /* backward solve the L^T */
  for ( i=n-1; i>=0; i-- ){
    v   = aa + aij->diag[i] - 2;
    vi  = aj + aij->diag[i] - 2;
    nz  = aij->diag[i] - ai[i];
    while (nz--) {
      tmp[*vi-- - 1] -= (*v--)*tmp[i];
    }
  }

  /* copy tmp into x according to permutation */
  for ( i=0; i<n; i++ ) x[r[i]] += tmp[i];

  ierr = ISRestoreIndices(invisrow,&r); CHKERRQ(ierr);
  ierr = ISRestoreIndices(inviscol,&c); CHKERRQ(ierr);
  ierr = ISDestroy(invisrow); CHKERRQ(ierr);
  ierr = ISDestroy(inviscol); CHKERRQ(ierr);

  PLogFlops(2*aij->nz);
  return 0;
}
/* ----------------------------------------------------------------*/
int MatILUFactorSymbolic_AIJ(Mat mat,IS isrow,IS iscol,double f,
                             int levels,Mat *fact)
{
  Mat_AIJ *aij = (Mat_AIJ *) mat->data, *aijnew;
  IS      isicol;
  int     *r,*ic, ierr, prow, n = aij->m, *ai = aij->i, *aj = aij->j;
  int     *ainew,*ajnew, jmax,*fill, *xi, nz, *im,*ajfill,*flev;
  int     *dloc, idx, row,m,fm, nzf, nzi,len,  realloc = 0;
  int     incrlev,nnz,i;
 
  if (n != aij->n) 
    SETERRQ(1,"MatILUFactorSymbolic_AIJ:Matrix must be square");
  if (!isrow) 
    SETERRQ(1,"MatILUFactorSymbolic_AIJ:Matrix must have row permutation");
  if (!iscol) SETERRQ(1,
    "MatILUFactorSymbolic_AIJ:Matrix must have column permutation");

  ierr = ISInvertPermutation(iscol,&isicol); CHKERRQ(ierr);
  ierr = ISGetIndices(isrow,&r); CHKERRQ(ierr);
  ierr = ISGetIndices(isicol,&ic); CHKERRQ(ierr);

  /* get new row pointers */
  ainew = (int *) PETSCMALLOC( (n+1)*sizeof(int) ); CHKPTRQ(ainew);
  ainew[0] = 1;
  /* don't know how many column pointers are needed so estimate */
  jmax = (int) (f*ai[n]);
  ajnew = (int *) PETSCMALLOC( (jmax)*sizeof(int) ); CHKPTRQ(ajnew);
  /* ajfill is level of fill for each fill entry */
  ajfill = (int *) PETSCMALLOC( (jmax)*sizeof(int) ); CHKPTRQ(ajfill);
  /* fill is a linked list of nonzeros in active row */
  fill = (int *) PETSCMALLOC( (n+1)*sizeof(int)); CHKPTRQ(fill);
  /* im is level for each filled value */
  im = (int *) PETSCMALLOC( (n+1)*sizeof(int)); CHKPTRQ(im);
  /* dloc is location of diagonal in factor */
  dloc = (int *) PETSCMALLOC( (n+1)*sizeof(int)); CHKPTRQ(dloc);
  dloc[0]  = 0;

  for ( prow=0; prow<n; prow++ ) {
    /* first copy previous fill into linked list */
    nzf = nz  = ai[r[prow]+1] - ai[r[prow]];
    xi  = aj + ai[r[prow]] - 1;
    fill[n] = n;
    while (nz--) {
      fm  = n;
      idx = ic[*xi++ - 1];
      do {
        m  = fm;
        fm = fill[m];
      } while (fm < idx);
      fill[m]   = idx;
      fill[idx] = fm;
      im[idx]   = 0;
    }
    nzi = 0;
    row = fill[n];
    while ( row < prow ) {
      incrlev = im[row] + 1;
      nz      = dloc[row];
      xi      = ajnew  + ainew[row] - 1 + nz;
      flev    = ajfill + ainew[row] - 1 + nz + 1;
      nnz     = ainew[row+1] - ainew[row] - nz - 1;
      if (*xi++ - 1 != row) {
        SETERRQ(1,"MatILUFactorSymbolic_AIJ:zero pivot");
      }
      fm      = row;
      while (nnz-- > 0) {
        idx = *xi++ - 1;
        if (*flev + incrlev > levels) {
          flev++;
          continue;
        }
        do {
          m  = fm;
          fm = fill[m];
        } while (fm < idx);
        if (fm != idx) {
          im[idx]  = *flev + incrlev;
          fill[m] = idx;
          fill[idx] = fm;
          fm = idx;
          nzf++;
        }
        else {
          if (im[idx] > *flev + incrlev) im[idx] = *flev+incrlev;
        }
        flev++;
      }
      row = fill[row];
      nzi++;
    }
    /* copy new filled row into permanent storage */
    ainew[prow+1] = ainew[prow] + nzf;
    if (ainew[prow+1] > jmax+1) {
      /* allocate a longer ajnew */
      int maxadd;
      maxadd = (int) ((f*ai[n]*(n-prow+5))/n);
      if (maxadd < nzf) maxadd = (n-prow)*(nzf+1);
      jmax += maxadd;
      xi = (int *) PETSCMALLOC( jmax*sizeof(int) );CHKPTRQ(xi);
      PETSCMEMCPY(xi,ajnew,(ainew[prow]-1)*sizeof(int));
      PETSCFREE(ajnew);
      ajnew = xi;
      /* allocate a longer ajfill */
      xi = (int *) PETSCMALLOC( jmax*sizeof(int) );CHKPTRQ(xi);
      PETSCMEMCPY(xi,ajfill,(ainew[prow]-1)*sizeof(int));
      PETSCFREE(ajfill);
      ajfill = xi;
      realloc++;
    }
    xi          = ajnew + ainew[prow] - 1;
    flev        = ajfill + ainew[prow] - 1;
    dloc[prow]  = nzi;
    fm          = fill[n];
    while (nzf--) {
      *xi++   = fm + 1;
      *flev++ = im[fm];
      fm      = fill[fm];
    }
  }
  PETSCFREE(ajfill); 
  ierr = ISRestoreIndices(isrow,&r); CHKERRQ(ierr);
  ierr = ISRestoreIndices(isicol,&ic); CHKERRQ(ierr);
  ierr = ISDestroy(isicol); CHKERRQ(ierr);
  PETSCFREE(fill); PETSCFREE(im);

  PLogInfo((PetscObject)mat,
    "Info:MatILUFactorSymbolic_AIJ:Realloc %d Fill ratio:given %g needed %g\n",
                             realloc,f,((double)ainew[n])/((double)ai[prow]));

  /* put together the new matrix */
  ierr = MatCreateSequentialAIJ(mat->comm,n, n, 0, 0, fact); CHKERRQ(ierr);
  aijnew = (Mat_AIJ *) (*fact)->data;
  PETSCFREE(aijnew->imax);
  aijnew->singlemalloc = 0;
  len = (ainew[n] - 1)*sizeof(Scalar);
  /* the next line frees the default space generated by the Create() */
  PETSCFREE(aijnew->a); PETSCFREE(aijnew->ilen);
  aijnew->a         = (Scalar *) PETSCMALLOC( len ); CHKPTRQ(aijnew->a);
  aijnew->j         = ajnew;
  aijnew->i         = ainew;
  for ( i=0; i<n; i++ ) dloc[i] += ainew[i];
  aijnew->diag      = dloc;
  aijnew->ilen      = 0;
  aijnew->imax      = 0;
  aijnew->row       = isrow;
  aijnew->col       = iscol;
  aijnew->solve_work = (Scalar *) PETSCMALLOC( (n+1)*sizeof(Scalar)); 
  CHKPTRQ(aijnew->solve_work);
  /* In aijnew structure:  Free imax, ilen, old a, old j.  
     Allocate dloc, solve_work, new a, new j */
  PLogObjectMemory(*fact,(ainew[n]-1-n) * (sizeof(int)+sizeof(Scalar)));
  aijnew->maxnz = aijnew->nz = ainew[n] - 1;
  (*fact)->factor   = FACTOR_LU;
  return 0; 
}
