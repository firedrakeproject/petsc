#include <../src/tao/complementarity/impls/ssls/ssls.h>
/*
   Context for ASXLS
     -- active-set      - reduced matrices formed
                          - inherit properties of original system
     -- semismooth (S)  - function not differentiable
                        - merit function continuously differentiable
                        - Fischer-Burmeister reformulation of complementarity
                          - Billups composition for two finite bounds
     -- infeasible (I)  - iterates not guaranteed to remain within bounds
     -- feasible (F)    - iterates guaranteed to remain within bounds
     -- linesearch (LS) - Armijo rule on direction

   Many other reformulations are possible and combinations of
   feasible/infeasible and linesearch/trust region are possible.

   Basic theory
     Fischer-Burmeister reformulation is semismooth with a continuously
     differentiable merit function and strongly semismooth if the F has
     lipschitz continuous derivatives.

     Every accumulation point generated by the algorithm is a stationary
     point for the merit function.  Stationary points of the merit function
     are solutions of the complementarity problem if
       a.  the stationary point has a BD-regular subdifferential, or
       b.  the Schur complement F'/F'_ff is a P_0-matrix where ff is the
           index set corresponding to the free variables.

     If one of the accumulation points has a BD-regular subdifferential then
       a.  the entire sequence converges to this accumulation point at
           a local q-superlinear rate
       b.  if in addition the reformulation is strongly semismooth near
           this accumulation point, then the algorithm converges at a
           local q-quadratic rate.

   The theory for the feasible version follows from the feasible descent
   algorithm framework. See {cite}`billups:algorithms`, {cite}`deluca.facchinei.ea:semismooth`,
   {cite}`ferris.kanzow.ea:feasible`, {cite}`fischer:special`, and {cite}`munson.facchinei.ea:semismooth`.
*/

static PetscErrorCode TaoSetUp_ASILS(Tao tao)
{
  TAO_SSLS *asls = (TAO_SSLS *)tao->data;

  PetscFunctionBegin;
  PetscCall(VecDuplicate(tao->solution, &tao->gradient));
  PetscCall(VecDuplicate(tao->solution, &tao->stepdirection));
  PetscCall(VecDuplicate(tao->solution, &asls->ff));
  PetscCall(VecDuplicate(tao->solution, &asls->dpsi));
  PetscCall(VecDuplicate(tao->solution, &asls->da));
  PetscCall(VecDuplicate(tao->solution, &asls->db));
  PetscCall(VecDuplicate(tao->solution, &asls->t1));
  PetscCall(VecDuplicate(tao->solution, &asls->t2));
  asls->fixed    = NULL;
  asls->free     = NULL;
  asls->J_sub    = NULL;
  asls->Jpre_sub = NULL;
  asls->w        = NULL;
  asls->r1       = NULL;
  asls->r2       = NULL;
  asls->r3       = NULL;
  asls->dxfree   = NULL;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode Tao_ASLS_FunctionGradient(TaoLineSearch ls, Vec X, PetscReal *fcn, Vec G, void *ptr)
{
  Tao       tao  = (Tao)ptr;
  TAO_SSLS *asls = (TAO_SSLS *)tao->data;

  PetscFunctionBegin;
  PetscCall(TaoComputeConstraints(tao, X, tao->constraints));
  PetscCall(VecFischer(X, tao->constraints, tao->XL, tao->XU, asls->ff));
  PetscCall(VecNorm(asls->ff, NORM_2, &asls->merit));
  *fcn = 0.5 * asls->merit * asls->merit;

  PetscCall(TaoComputeJacobian(tao, tao->solution, tao->jacobian, tao->jacobian_pre));
  PetscCall(MatDFischer(tao->jacobian, tao->solution, tao->constraints, tao->XL, tao->XU, asls->t1, asls->t2, asls->da, asls->db));
  PetscCall(VecPointwiseMult(asls->t1, asls->ff, asls->db));
  PetscCall(MatMultTranspose(tao->jacobian, asls->t1, G));
  PetscCall(VecPointwiseMult(asls->t1, asls->ff, asls->da));
  PetscCall(VecAXPY(G, 1.0, asls->t1));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TaoDestroy_ASILS(Tao tao)
{
  TAO_SSLS *ssls = (TAO_SSLS *)tao->data;

  PetscFunctionBegin;
  PetscCall(VecDestroy(&ssls->ff));
  PetscCall(VecDestroy(&ssls->dpsi));
  PetscCall(VecDestroy(&ssls->da));
  PetscCall(VecDestroy(&ssls->db));
  PetscCall(VecDestroy(&ssls->w));
  PetscCall(VecDestroy(&ssls->t1));
  PetscCall(VecDestroy(&ssls->t2));
  PetscCall(VecDestroy(&ssls->r1));
  PetscCall(VecDestroy(&ssls->r2));
  PetscCall(VecDestroy(&ssls->r3));
  PetscCall(VecDestroy(&ssls->dxfree));
  PetscCall(MatDestroy(&ssls->J_sub));
  PetscCall(MatDestroy(&ssls->Jpre_sub));
  PetscCall(ISDestroy(&ssls->fixed));
  PetscCall(ISDestroy(&ssls->free));
  PetscCall(KSPDestroy(&tao->ksp));
  PetscCall(PetscFree(tao->data));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TaoSolve_ASILS(Tao tao)
{
  TAO_SSLS                    *asls = (TAO_SSLS *)tao->data;
  PetscReal                    psi, ndpsi, normd, innerd, t = 0;
  PetscInt                     nf;
  TaoLineSearchConvergedReason ls_reason;

  PetscFunctionBegin;
  /* Assume that Setup has been called!
     Set the structure for the Jacobian and create a linear solver. */

  PetscCall(TaoComputeVariableBounds(tao));
  PetscCall(TaoLineSearchSetObjectiveAndGradientRoutine(tao->linesearch, Tao_ASLS_FunctionGradient, tao));
  PetscCall(TaoLineSearchSetObjectiveRoutine(tao->linesearch, Tao_SSLS_Function, tao));

  /* Calculate the function value and fischer function value at the
     current iterate */
  PetscCall(TaoLineSearchComputeObjectiveAndGradient(tao->linesearch, tao->solution, &psi, asls->dpsi));
  PetscCall(VecNorm(asls->dpsi, NORM_2, &ndpsi));

  tao->reason = TAO_CONTINUE_ITERATING;
  while (1) {
    /* Check the termination criteria */
    PetscCall(PetscInfo(tao, "iter %" PetscInt_FMT ", merit: %g, ||dpsi||: %g\n", tao->niter, (double)asls->merit, (double)ndpsi));
    PetscCall(TaoLogConvergenceHistory(tao, asls->merit, ndpsi, 0.0, tao->ksp_its));
    PetscCall(TaoMonitor(tao, tao->niter, asls->merit, ndpsi, 0.0, t));
    PetscUseTypeMethod(tao, convergencetest, tao->cnvP);
    if (TAO_CONTINUE_ITERATING != tao->reason) break;

    /* Call general purpose update function */
    PetscTryTypeMethod(tao, update, tao->niter, tao->user_update);
    tao->niter++;

    /* We are going to solve a linear system of equations.  We need to
       set the tolerances for the solve so that we maintain an asymptotic
       rate of convergence that is superlinear.
       Note: these tolerances are for the reduced system.  We really need
       to make sure that the full system satisfies the full-space conditions.

       This rule gives superlinear asymptotic convergence
       asls->atol = min(0.5, asls->merit*sqrt(asls->merit));
       asls->rtol = 0.0;

       This rule gives quadratic asymptotic convergence
       asls->atol = min(0.5, asls->merit*asls->merit);
       asls->rtol = 0.0;

       Calculate a free and fixed set of variables.  The fixed set of
       variables are those for the d_b is approximately equal to zero.
       The definition of approximately changes as we approach the solution
       to the problem.

       No one rule is guaranteed to work in all cases.  The following
       definition is based on the norm of the Jacobian matrix.  If the
       norm is large, the tolerance becomes smaller. */
    PetscCall(MatNorm(tao->jacobian, NORM_1, &asls->identifier));
    asls->identifier = PetscMin(asls->merit, 1e-2) / (1 + asls->identifier);

    PetscCall(VecSet(asls->t1, -asls->identifier));
    PetscCall(VecSet(asls->t2, asls->identifier));

    PetscCall(ISDestroy(&asls->fixed));
    PetscCall(ISDestroy(&asls->free));
    PetscCall(VecWhichBetweenOrEqual(asls->t1, asls->db, asls->t2, &asls->fixed));
    PetscCall(ISComplementVec(asls->fixed, asls->t1, &asls->free));

    PetscCall(ISGetSize(asls->fixed, &nf));
    PetscCall(PetscInfo(tao, "Number of fixed variables: %" PetscInt_FMT "\n", nf));

    /* We now have our partition.  Now calculate the direction in the
       fixed variable space. */
    PetscCall(TaoVecGetSubVec(asls->ff, asls->fixed, tao->subset_type, 0.0, &asls->r1));
    PetscCall(TaoVecGetSubVec(asls->da, asls->fixed, tao->subset_type, 1.0, &asls->r2));
    PetscCall(VecPointwiseDivide(asls->r1, asls->r1, asls->r2));
    PetscCall(VecSet(tao->stepdirection, 0.0));
    PetscCall(VecISAXPY(tao->stepdirection, asls->fixed, 1.0, asls->r1));

    /* Our direction in the Fixed Variable Set is fixed.  Calculate the
       information needed for the step in the Free Variable Set.  To
       do this, we need to know the diagonal perturbation and the
       right-hand side. */

    PetscCall(TaoVecGetSubVec(asls->da, asls->free, tao->subset_type, 0.0, &asls->r1));
    PetscCall(TaoVecGetSubVec(asls->ff, asls->free, tao->subset_type, 0.0, &asls->r2));
    PetscCall(TaoVecGetSubVec(asls->db, asls->free, tao->subset_type, 1.0, &asls->r3));
    PetscCall(VecPointwiseDivide(asls->r1, asls->r1, asls->r3));
    PetscCall(VecPointwiseDivide(asls->r2, asls->r2, asls->r3));

    /* r1 is the diagonal perturbation
       r2 is the right-hand side
       r3 is no longer needed

       Now need to modify r2 for our direction choice in the fixed
       variable set:  calculate t1 = J*d, take the reduced vector
       of t1 and modify r2. */

    PetscCall(MatMult(tao->jacobian, tao->stepdirection, asls->t1));
    PetscCall(TaoVecGetSubVec(asls->t1, asls->free, tao->subset_type, 0.0, &asls->r3));
    PetscCall(VecAXPY(asls->r2, -1.0, asls->r3));

    /* Calculate the reduced problem matrix and the direction */
    if (!asls->w && (tao->subset_type == TAO_SUBSET_MASK || tao->subset_type == TAO_SUBSET_MATRIXFREE)) PetscCall(VecDuplicate(tao->solution, &asls->w));
    PetscCall(TaoMatGetSubMat(tao->jacobian, asls->free, asls->w, tao->subset_type, &asls->J_sub));
    if (tao->jacobian != tao->jacobian_pre) {
      PetscCall(TaoMatGetSubMat(tao->jacobian_pre, asls->free, asls->w, tao->subset_type, &asls->Jpre_sub));
    } else {
      PetscCall(MatDestroy(&asls->Jpre_sub));
      asls->Jpre_sub = asls->J_sub;
      PetscCall(PetscObjectReference((PetscObject)asls->Jpre_sub));
    }
    PetscCall(MatDiagonalSet(asls->J_sub, asls->r1, ADD_VALUES));
    PetscCall(TaoVecGetSubVec(tao->stepdirection, asls->free, tao->subset_type, 0.0, &asls->dxfree));
    PetscCall(VecSet(asls->dxfree, 0.0));

    /* Calculate the reduced direction.  (Really negative of Newton
       direction.  Therefore, rest of the code uses -d.) */
    PetscCall(KSPReset(tao->ksp));
    PetscCall(KSPSetOperators(tao->ksp, asls->J_sub, asls->Jpre_sub));
    PetscCall(KSPSolve(tao->ksp, asls->r2, asls->dxfree));
    PetscCall(KSPGetIterationNumber(tao->ksp, &tao->ksp_its));
    tao->ksp_tot_its += tao->ksp_its;

    /* Add the direction in the free variables back into the real direction. */
    PetscCall(VecISAXPY(tao->stepdirection, asls->free, 1.0, asls->dxfree));

    /* Check the real direction for descent and if not, use the negative
       gradient direction. */
    PetscCall(VecNorm(tao->stepdirection, NORM_2, &normd));
    PetscCall(VecDot(tao->stepdirection, asls->dpsi, &innerd));

    if (innerd <= asls->delta * PetscPowReal(normd, asls->rho)) {
      PetscCall(PetscInfo(tao, "Gradient direction: %5.4e.\n", (double)innerd));
      PetscCall(PetscInfo(tao, "Iteration %" PetscInt_FMT ": newton direction not descent\n", tao->niter));
      PetscCall(VecCopy(asls->dpsi, tao->stepdirection));
      PetscCall(VecDot(asls->dpsi, tao->stepdirection, &innerd));
    }

    PetscCall(VecScale(tao->stepdirection, -1.0));
    innerd = -innerd;

    /* We now have a correct descent direction.  Apply a linesearch to
       find the new iterate. */
    PetscCall(TaoLineSearchSetInitialStepLength(tao->linesearch, 1.0));
    PetscCall(TaoLineSearchApply(tao->linesearch, tao->solution, &psi, asls->dpsi, tao->stepdirection, &t, &ls_reason));
    PetscCall(VecNorm(asls->dpsi, NORM_2, &ndpsi));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*MC
   TAOASILS - Active-set infeasible linesearch algorithm for solving complementarity constraints

   Options Database Keys:
+ -tao_ssls_delta - descent test fraction
- -tao_ssls_rho   - descent test power

  Level: beginner

  Note:
  See {cite}`billups:algorithms`, {cite}`deluca.facchinei.ea:semismooth`,
  {cite}`ferris.kanzow.ea:feasible`, {cite}`fischer:special`, and {cite}`munson.facchinei.ea:semismooth`.

.seealso: `Tao`, `TaoType`, `TAOASFLS`
M*/
PETSC_EXTERN PetscErrorCode TaoCreate_ASILS(Tao tao)
{
  TAO_SSLS   *asls;
  const char *armijo_type = TAOLINESEARCHARMIJO;

  PetscFunctionBegin;
  PetscCall(PetscNew(&asls));
  tao->data                = (void *)asls;
  tao->ops->solve          = TaoSolve_ASILS;
  tao->ops->setup          = TaoSetUp_ASILS;
  tao->ops->view           = TaoView_SSLS;
  tao->ops->setfromoptions = TaoSetFromOptions_SSLS;
  tao->ops->destroy        = TaoDestroy_ASILS;
  tao->subset_type         = TAO_SUBSET_SUBVEC;
  asls->delta              = 1e-10;
  asls->rho                = 2.1;
  asls->fixed              = NULL;
  asls->free               = NULL;
  asls->J_sub              = NULL;
  asls->Jpre_sub           = NULL;
  asls->w                  = NULL;
  asls->r1                 = NULL;
  asls->r2                 = NULL;
  asls->r3                 = NULL;
  asls->t1                 = NULL;
  asls->t2                 = NULL;
  asls->dxfree             = NULL;

  asls->identifier = 1e-5;

  PetscCall(TaoParametersInitialize(tao));

  PetscCall(TaoLineSearchCreate(((PetscObject)tao)->comm, &tao->linesearch));
  PetscCall(PetscObjectIncrementTabLevel((PetscObject)tao->linesearch, (PetscObject)tao, 1));
  PetscCall(TaoLineSearchSetType(tao->linesearch, armijo_type));
  PetscCall(TaoLineSearchSetOptionsPrefix(tao->linesearch, tao->hdr.prefix));
  PetscCall(TaoLineSearchSetFromOptions(tao->linesearch));

  PetscCall(KSPCreate(((PetscObject)tao)->comm, &tao->ksp));
  PetscCall(PetscObjectIncrementTabLevel((PetscObject)tao->ksp, (PetscObject)tao, 1));
  PetscCall(KSPSetOptionsPrefix(tao->ksp, tao->hdr.prefix));
  PetscCall(KSPSetFromOptions(tao->ksp));

  /* Override default settings (unless already changed) */
  PetscObjectParameterSetDefault(tao, max_it, 2000);
  PetscObjectParameterSetDefault(tao, max_funcs, 4000);
  PetscObjectParameterSetDefault(tao, gttol, 0);
  PetscObjectParameterSetDefault(tao, grtol, 0);
  PetscObjectParameterSetDefault(tao, gatol, PetscDefined(USE_REAL_SINGLE) ? 1.0e-6 : 1.0e-16);
  PetscObjectParameterSetDefault(tao, fmin, PetscDefined(USE_REAL_SINGLE) ? 1.0e-4 : 1.0e-8);
  PetscFunctionReturn(PETSC_SUCCESS);
}
