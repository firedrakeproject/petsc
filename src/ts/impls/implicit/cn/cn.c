/*$Id: cn.c,v 1.28 2001/03/28 19:42:32 balay Exp balay $*/
/*
       Code for Timestepping with implicit Crank-Nicholson method.
    THIS IS NOT YET COMPLETE -- DO NOT USE!!
*/
#include "src/ts/tsimpl.h"                /*I   "petscts.h"   I*/

typedef struct {
  Vec  update;      /* work vector where new solution is formed */
  Vec  func;        /* work vector where F(t[i],u[i]) is stored */
  Vec  rhs;         /* work vector for RHS; vec_sol/dt */
} TS_CN;

/*------------------------------------------------------------------------------*/
#undef __FUNCT__  
#define __FUNCT__ "TSComputeRHSFunctionEuler"
/*
   TSComputeRHSFunctionEuler - Evaluates the right-hand-side function. 

   Note: If the user did not provide a function but merely a matrix,
   this routine applies the matrix.
*/
int TSComputeRHSFunctionEuler(TS ts,double t,Vec x,Vec y)
{
  int    ierr;
  Scalar neg_two = -2.0,neg_mdt = -1.0/ts->time_step;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_COOKIE);
  PetscValidHeader(x);  PetscValidHeader(y);

  if (ts->rhsfunction) {
    PetscStackPush("TS user right-hand-side function");
    ierr = (*ts->rhsfunction)(ts,t,x,y,ts->funP);CHKERRQ(ierr);
    PetscStackPop;
    PetscFunctionReturn(0);
  }

  if (ts->rhsmatrix) { /* assemble matrix for this timestep */
    MatStructure flg;
    PetscStackPush("TS user right-hand-side matrix function");
    ierr = (*ts->rhsmatrix)(ts,t,&ts->A,&ts->B,&flg,ts->jacP);CHKERRQ(ierr);
    PetscStackPop;
  }
  ierr = MatMult(ts->A,x,y);CHKERRQ(ierr);
  /* shift: y = y -2*x */
  ierr = VecAXPY(&neg_two,x,y);CHKERRQ(ierr);
  /* scale: y = y -2*x */
  ierr = VecScale(&neg_mdt,y);CHKERRQ(ierr);

  /* apply user-provided boundary conditions (only needed if these are time dependent) */
  ierr = TSComputeRHSBoundaryConditions(ts,t,y);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

/*
    Version for linear PDE where RHS does not depend on time. Has built a
  single matrix that is to be used for all timesteps.
*/
#undef __FUNCT__  
#define __FUNCT__ "TSStep_CN_Linear_Constant_Matrix"
static int TSStep_CN_Linear_Constant_Matrix(TS ts,int *steps,double *ptime)
{
  TS_CN     *cn = (TS_CN*)ts->data;
  Vec       sol = ts->vec_sol,update = cn->update;
  Vec       rhs = cn->rhs;
  int       ierr,i,max_steps = ts->max_steps,its;
  Scalar    dt = ts->time_step,two = 2.0;
  
  PetscFunctionBegin;
  *steps = -ts->steps;
  ierr = TSMonitor(ts,ts->steps,ts->ptime,sol);CHKERRQ(ierr);

  /* set initial guess to be previous solution */
  ierr = VecCopy(sol,update);CHKERRQ(ierr);

  for (i=0; i<max_steps; i++) {
    ts->ptime += ts->time_step;
    if (ts->ptime > ts->max_time) break;

    /* phase 1 - explicit step */
    ierr = TSComputeRHSFunctionEuler(ts,ts->ptime,sol,update);CHKERRQ(ierr);
    ierr = VecAXPBY(&dt,&two,update,sol);CHKERRQ(ierr);

    /* phase 2 - implicit step */
    ierr = VecCopy(sol,rhs);CHKERRQ(ierr);
    /* apply user-provided boundary conditions (only needed if they are time dependent) */
    ierr = TSComputeRHSBoundaryConditions(ts,ts->ptime,rhs);CHKERRQ(ierr);

    ierr = SLESSolve(ts->sles,rhs,update,&its);CHKERRQ(ierr);
    ts->linear_its += PetscAbsInt(its);
    ierr = VecCopy(update,sol);CHKERRQ(ierr);
    ts->steps++;
    ierr = TSMonitor(ts,ts->steps,ts->ptime,sol);CHKERRQ(ierr);
  }

  *steps += ts->steps;
  *ptime  = ts->ptime;
  PetscFunctionReturn(0);
}
/*
      Version where matrix depends on time 
*/
#undef __FUNCT__  
#define __FUNCT__ "TSStep_CN_Linear_Variable_Matrix"
static int TSStep_CN_Linear_Variable_Matrix(TS ts,int *steps,double *ptime)
{
  TS_CN        *cn = (TS_CN*)ts->data;
  Vec          sol = ts->vec_sol,update = cn->update,rhs = cn->rhs;
  int          ierr,i,max_steps = ts->max_steps,its;
  Scalar       dt = ts->time_step,two = 2.0,neg_dt = -1.0*ts->time_step;
  MatStructure str;

  PetscFunctionBegin;
  *steps = -ts->steps;
  ierr = TSMonitor(ts,ts->steps,ts->ptime,sol);CHKERRQ(ierr);

  /* set initial guess to be previous solution */
  ierr = VecCopy(sol,update);CHKERRQ(ierr);

  for (i=0; i<max_steps; i++) {
    ts->ptime += ts->time_step;
    if (ts->ptime > ts->max_time) break;
    /*
        evaluate matrix function 
    */
    ierr = (*ts->rhsmatrix)(ts,ts->ptime,&ts->A,&ts->B,&str,ts->jacP);CHKERRQ(ierr);
    if (!ts->Ashell) {
      ierr = MatScale(&neg_dt,ts->A);CHKERRQ(ierr);
      ierr = MatShift(&two,ts->A);CHKERRQ(ierr);
    }
    if (ts->B != ts->A && ts->Ashell != ts->B && str != SAME_PRECONDITIONER) {
      ierr = MatScale(&neg_dt,ts->B);CHKERRQ(ierr);
      ierr = MatShift(&two,ts->B);CHKERRQ(ierr);
    }

    /* phase 1 - explicit step */
    ierr = TSComputeRHSFunctionEuler(ts,ts->ptime,sol,update);CHKERRQ(ierr);
    ierr = VecAXPBY(&dt,&two,update,sol);CHKERRQ(ierr);

    /* phase 2 - implicit step */
    ierr = VecCopy(sol,rhs);CHKERRQ(ierr);

    /* apply user-provided boundary conditions (only needed if they are time dependent) */
    ierr = TSComputeRHSBoundaryConditions(ts,ts->ptime,rhs);CHKERRQ(ierr);

    ierr = SLESSetOperators(ts->sles,ts->A,ts->B,str);CHKERRQ(ierr);
    ierr = SLESSolve(ts->sles,rhs,update,&its);CHKERRQ(ierr);
    ts->linear_its += PetscAbsInt(its);
    ierr = VecCopy(update,sol);CHKERRQ(ierr);
    ts->steps++;
    ierr = TSMonitor(ts,ts->steps,ts->ptime,sol);CHKERRQ(ierr);
  }

  *steps += ts->steps;
  *ptime  = ts->ptime;
  PetscFunctionReturn(0);
}
/*
    Version for nonlinear PDE.
*/
#undef __FUNCT__  
#define __FUNCT__ "TSStep_CN_Nonlinear"
static int TSStep_CN_Nonlinear(TS ts,int *steps,double *ptime)
{
  Vec       sol = ts->vec_sol;
  int       ierr,i,max_steps = ts->max_steps,its,lits;
  TS_CN *cn = (TS_CN*)ts->data;
  
  PetscFunctionBegin;
  *steps = -ts->steps;
  ierr = TSMonitor(ts,ts->steps,ts->ptime,sol);CHKERRQ(ierr);

  for (i=0; i<max_steps; i++) {
    ts->ptime += ts->time_step;
    if (ts->ptime > ts->max_time) break;
    ierr = VecCopy(sol,cn->update);CHKERRQ(ierr);
    ierr = SNESSolve(ts->snes,cn->update,&its);CHKERRQ(ierr);
    ierr = SNESGetNumberLinearIterations(ts->snes,&lits);CHKERRQ(ierr);
    ts->nonlinear_its += PetscAbsInt(its); ts->linear_its += lits;
    ierr = VecCopy(cn->update,sol);CHKERRQ(ierr);
    ts->steps++;
    ierr = TSMonitor(ts,ts->steps,ts->ptime,sol);CHKERRQ(ierr);
  }

  *steps += ts->steps;
  *ptime  = ts->ptime;
  PetscFunctionReturn(0);
}

/*------------------------------------------------------------*/
#undef __FUNCT__  
#define __FUNCT__ "TSDestroy_CN"
static int TSDestroy_CN(TS ts)
{
  TS_CN *cn = (TS_CN*)ts->data;
  int       ierr;

  PetscFunctionBegin;
  if (cn->update) {ierr = VecDestroy(cn->update);CHKERRQ(ierr);}
  if (cn->func) {ierr = VecDestroy(cn->func);CHKERRQ(ierr);}
  if (cn->rhs) {ierr = VecDestroy(cn->rhs);CHKERRQ(ierr);}
  if (ts->Ashell) {ierr = MatDestroy(ts->A);CHKERRQ(ierr);}
  ierr = PetscFree(cn);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


/*------------------------------------------------------------*/
/*
    This matrix shell multiply where user provided Shell matrix
*/

#undef __FUNCT__  
#define __FUNCT__ "TSCnMatMult"
int TSCnMatMult(Mat mat,Vec x,Vec y)
{
  TS     ts;
  Scalar two = 2.0,neg_dt;
  int    ierr;

  PetscFunctionBegin;
  ierr = MatShellGetContext(mat,(void **)&ts);CHKERRQ(ierr);
  neg_dt = -1.0*ts->time_step;

  /* apply user-provided function */
  ierr = MatMult(ts->Ashell,x,y);CHKERRQ(ierr);
  /* shift and scale by 2 - dt*F */
  ierr = VecAXPBY(&two,&neg_dt,x,y);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* 
    This defines the nonlinear equation that is to be solved with SNES

              U^{n+1} - dt*F(U^{n+1}) - U^{n}
*/
#undef __FUNCT__  
#define __FUNCT__ "TSCnFunction"
int TSCnFunction(SNES snes,Vec x,Vec y,void *ctx)
{
  TS     ts = (TS) ctx;
  Scalar mdt = 1.0/ts->time_step,*unp1,*un,*Funp1;
  int    ierr,i,n;

  PetscFunctionBegin;
  /* apply user provided function */
  ierr = TSComputeRHSFunction(ts,ts->ptime,x,y);CHKERRQ(ierr);
  /* (u^{n+1} - U^{n})/dt - F(u^{n+1}) */
  ierr = VecGetArray(ts->vec_sol,&un);CHKERRQ(ierr);
  ierr = VecGetArray(x,&unp1);CHKERRQ(ierr);
  ierr = VecGetArray(y,&Funp1);CHKERRQ(ierr);
  ierr = VecGetLocalSize(x,&n);CHKERRQ(ierr);

  for (i=0; i<n; i++) {
    Funp1[i] = mdt*(unp1[i] - un[i]) - Funp1[i];
  }
  ierr = VecRestoreArray(ts->vec_sol,&un);CHKERRQ(ierr);
  ierr = VecRestoreArray(x,&unp1);CHKERRQ(ierr);
  ierr = VecRestoreArray(y,&Funp1);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*
   This constructs the Jacobian needed for SNES 

             J = I/dt - J_{F}   where J_{F} is the given Jacobian of F.
*/
#undef __FUNCT__  
#define __FUNCT__ "TSCnJacobian"
int TSCnJacobian(SNES snes,Vec x,Mat *AA,Mat *BB,MatStructure *str,void *ctx)
{
  TS         ts = (TS) ctx;
  int        ierr;
  Scalar     mone = -1.0,mdt = 1.0/ts->time_step;
  PetscTruth isshell;

  PetscFunctionBegin;
  /* construct user's Jacobian */
  ierr = TSComputeRHSJacobian(ts,ts->ptime,x,AA,BB,str);CHKERRQ(ierr);

  /* shift and scale Jacobian, if not matrix-free */
  ierr = PetscTypeCompare((PetscObject)*AA,MATSHELL,&isshell);CHKERRQ(ierr);
  if (!isshell) {
    ierr = MatScale(&mone,*AA);CHKERRQ(ierr);
    ierr = MatShift(&mdt,*AA);CHKERRQ(ierr);
  }
  ierr = PetscTypeCompare((PetscObject)*BB,MATSHELL,&isshell);CHKERRQ(ierr);
  if (*BB != *AA && *str != SAME_PRECONDITIONER && !isshell) {
    ierr = MatScale(&mone,*BB);CHKERRQ(ierr);
    ierr = MatShift(&mdt,*BB);CHKERRQ(ierr);
  }

  PetscFunctionReturn(0);
}

/* ------------------------------------------------------------*/
#undef __FUNCT__  
#define __FUNCT__ "TSSetUp_CN_Linear_Constant_Matrix"
static int TSSetUp_CN_Linear_Constant_Matrix(TS ts)
{
  TS_CN   *cn = (TS_CN*)ts->data;
  int     ierr,M,m;
  Scalar  two = 2.0,neg_dt = -1.0*ts->time_step;

  PetscFunctionBegin;
  ierr = VecDuplicate(ts->vec_sol,&cn->update);CHKERRQ(ierr);  
  ierr = VecDuplicate(ts->vec_sol,&cn->rhs);CHKERRQ(ierr);  
    
  /* build linear system to be solved */
  if (!ts->Ashell) {
    ierr = MatScale(&neg_dt,ts->A);CHKERRQ(ierr);
    ierr = MatShift(&two,ts->A);CHKERRQ(ierr);
  } else {
    /* construct new shell matrix */
    ierr = VecGetSize(ts->vec_sol,&M);CHKERRQ(ierr);
    ierr = VecGetLocalSize(ts->vec_sol,&m);CHKERRQ(ierr);
    ierr = MatCreateShell(ts->comm,m,M,M,M,ts,&ts->A);CHKERRQ(ierr);
    ierr = MatShellSetOperation(ts->A,MATOP_MULT,(void(*)())TSCnMatMult);CHKERRQ(ierr);
  }
  if (ts->A != ts->B && ts->Ashell != ts->B) {
    ierr = MatScale(&neg_dt,ts->B);CHKERRQ(ierr);
    ierr = MatShift(&two,ts->B);CHKERRQ(ierr);
  }
  ierr = SLESSetOperators(ts->sles,ts->A,ts->B,SAME_NONZERO_PATTERN);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "TSSetUp_CN_Linear_Variable_Matrix"
static int TSSetUp_CN_Linear_Variable_Matrix(TS ts)
{
  TS_CN *cn = (TS_CN*)ts->data;
  int       ierr,M,m;

  PetscFunctionBegin;
  ierr = VecDuplicate(ts->vec_sol,&cn->update);CHKERRQ(ierr);  
  ierr = VecDuplicate(ts->vec_sol,&cn->rhs);CHKERRQ(ierr);  
  if (ts->Ashell) { /* construct new shell matrix */
    ierr = VecGetSize(ts->vec_sol,&M);CHKERRQ(ierr);
    ierr = VecGetLocalSize(ts->vec_sol,&m);CHKERRQ(ierr);
    ierr = MatCreateShell(ts->comm,m,M,M,M,ts,&ts->A);CHKERRQ(ierr);
    ierr = MatShellSetOperation(ts->A,MATOP_MULT,(void(*)())TSCnMatMult);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "TSSetUp_CN_Nonlinear"
static int TSSetUp_CN_Nonlinear(TS ts)
{
  TS_CN *cn = (TS_CN*)ts->data;
  int       ierr,M,m;

  PetscFunctionBegin;
  ierr = VecDuplicate(ts->vec_sol,&cn->update);CHKERRQ(ierr);  
  ierr = VecDuplicate(ts->vec_sol,&cn->func);CHKERRQ(ierr);  
  ierr = SNESSetFunction(ts->snes,cn->func,TSCnFunction,ts);CHKERRQ(ierr);
  if (ts->Ashell) { /* construct new shell matrix */
    ierr = VecGetSize(ts->vec_sol,&M);CHKERRQ(ierr);
    ierr = VecGetLocalSize(ts->vec_sol,&m);CHKERRQ(ierr);
    ierr = MatCreateShell(ts->comm,m,M,M,M,ts,&ts->A);CHKERRQ(ierr);
    ierr = MatShellSetOperation(ts->A,MATOP_MULT,(void(*)())TSCnMatMult);CHKERRQ(ierr);
  }
  ierr = SNESSetJacobian(ts->snes,ts->A,ts->B,TSCnJacobian,ts);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
/*------------------------------------------------------------*/

#undef __FUNCT__  
#define __FUNCT__ "TSSetFromOptions_CN_Linear"
static int TSSetFromOptions_CN_Linear(TS ts)
{
  int ierr;

  PetscFunctionBegin;
  ierr = SLESSetFromOptions(ts->sles);CHKERRQ(ierr);
  
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "TSSetFromOptions_CN_Nonlinear"
static int TSSetFromOptions_CN_Nonlinear(TS ts)
{
  int ierr;

  PetscFunctionBegin;
  ierr = SNESSetFromOptions(ts->snes);CHKERRQ(ierr);
  
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "TSView_CN"
static int TSView_CN(TS ts,PetscViewer viewer)
{
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}

/* ------------------------------------------------------------ */
EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "TSCreate_CN"
int TSCreate_CN(TS ts)
{
  TS_CN      *cn;
  int        ierr;
  KSP        ksp;
  PetscTruth isshell;

  PetscFunctionBegin;
  ts->destroy         = TSDestroy_CN;
  ts->view            = TSView_CN;

  if (ts->problem_type == TS_LINEAR) {
    if (!ts->A) {
      SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"Must set rhs matrix for linear problem");
    }
    ierr = PetscTypeCompare((PetscObject)ts->A,MATSHELL,&isshell);CHKERRQ(ierr);
    if (!ts->rhsmatrix) {
      if (isshell) {
        ts->Ashell = ts->A;
      }
      ts->setup  = TSSetUp_CN_Linear_Constant_Matrix;
      ts->step   = TSStep_CN_Linear_Constant_Matrix;
    } else {
      if (isshell) {
        ts->Ashell = ts->A;
      }
      ts->setup  = TSSetUp_CN_Linear_Variable_Matrix;  
      ts->step   = TSStep_CN_Linear_Variable_Matrix;
    }
    ts->setfromoptions  = TSSetFromOptions_CN_Linear;
    ierr = SLESCreate(ts->comm,&ts->sles);CHKERRQ(ierr);
    ierr = SLESGetKSP(ts->sles,&ksp);CHKERRQ(ierr);
    ierr = KSPSetInitialGuessNonzero(ksp);CHKERRQ(ierr);
  } else if (ts->problem_type == TS_NONLINEAR) {
    ierr = PetscTypeCompare((PetscObject)ts->A,MATSHELL,&isshell);CHKERRQ(ierr);
    if (isshell) {
      ts->Ashell = ts->A;
    }
    ts->setup           = TSSetUp_CN_Nonlinear;  
    ts->step            = TSStep_CN_Nonlinear;
    ts->setfromoptions  = TSSetFromOptions_CN_Nonlinear;
    ierr = SNESCreate(ts->comm,SNES_NONLINEAR_EQUATIONS,&ts->snes);CHKERRQ(ierr);
  } else SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,"No such problem");

  ierr = PetscNew(TS_CN,&cn);CHKERRQ(ierr);
  PetscLogObjectMemory(ts,sizeof(TS_CN));
  ierr     = PetscMemzero(cn,sizeof(TS_CN));CHKERRQ(ierr);
  ts->data = (void*)cn;

  PetscFunctionReturn(0);
}
EXTERN_C_END





