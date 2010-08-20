#define PETSCSNES_DLL

#include "private/snesimpl.h"     /*I  "petscsnes.h"  I*/

EXTERN_C_BEGIN
EXTERN PetscErrorCode PETSCSNES_DLLEXPORT SNESCreate_LS(SNES);
EXTERN PetscErrorCode PETSCSNES_DLLEXPORT SNESCreate_TR(SNES);
EXTERN PetscErrorCode PETSCSNES_DLLEXPORT SNESCreate_Test(SNES);
EXTERN PetscErrorCode PETSCSNES_DLLEXPORT SNESCreate_Picard(SNES);
EXTERN PetscErrorCode PETSCSNES_DLLEXPORT SNESCreate_KSPONLY(SNES);
#if !defined(PETSC_USE_COMPLEX)
EXTERN PetscErrorCode PETSCSNES_DLLEXPORT SNESCreate_LSVI(SNES);
#endif
EXTERN_C_END

const char *SNESConvergedReasons_Shifted[]  = {" "," ","DIVERGED_LOCAL_MIN"," ","DIVERGED_LS_FAILURE","DIVERGED_MAX_IT",
                                               "DIVERGED_FNORM_NAN","DIVERGED_LINEAR_SOLVE","DIVERGED_FUNCTION_COUNT","DIVERGED_FUNCTION_DOMAIN",
                                               "CONVERGED_ITERATING"," ","CONVERGED_FNORM_ABS","CONVERGED_FNORM_RELATIVE",
                                               "CONVERGED_PNORM_RELATIVE","CONVERGED_ITS"," ","CONVERGED_TR_DELTA","SNESConvergedReason","",0};
const char **SNESConvergedReasons = SNESConvergedReasons_Shifted + 10;

/*
      This is used by SNESSetType() to make sure that at least one 
    SNESRegisterAll() is called. In general, if there is more than one
    DLL then SNESRegisterAll() may be called several times.
*/
extern PetscTruth SNESRegisterAllCalled;

#undef __FUNCT__  
#define __FUNCT__ "SNESRegisterAll"
/*@C
   SNESRegisterAll - Registers all of the nonlinear solver methods in the SNES package.

   Not Collective

   Level: advanced

.keywords: SNES, register, all

.seealso:  SNESRegisterDestroy()
@*/
PetscErrorCode PETSCSNES_DLLEXPORT SNESRegisterAll(const char path[])
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  SNESRegisterAllCalled = PETSC_TRUE;

  ierr = SNESRegisterDynamic(SNESLS,     path,"SNESCreate_LS",     SNESCreate_LS);CHKERRQ(ierr);
  ierr = SNESRegisterDynamic(SNESTR,     path,"SNESCreate_TR",     SNESCreate_TR);CHKERRQ(ierr);
  ierr = SNESRegisterDynamic(SNESTEST,   path,"SNESCreate_Test",   SNESCreate_Test);CHKERRQ(ierr);
  ierr = SNESRegisterDynamic(SNESPICARD, path,"SNESCreate_Picard", SNESCreate_Picard);CHKERRQ(ierr);
  ierr = SNESRegisterDynamic(SNESKSPONLY,path,"SNESCreate_KSPONLY",SNESCreate_KSPONLY);CHKERRQ(ierr);
#if !defined(PETSC_USE_COMPLEX)
  ierr = SNESRegisterDynamic(SNESLSVI,   path,"SNESCreate_LSVI",   SNESCreate_LSVI);CHKERRQ(ierr);
#endif
  PetscFunctionReturn(0);
}

