#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: psleep.c,v 1.11 1997/10/19 03:23:45 bsmith Exp bsmith $";
#endif
/*
     Provides utility routines for manulating any type of PETSc object.
*/
#include "petsc.h"  /*I   "petsc.h"    I*/

#if defined (PARCH_nt)
#include <stdlib.h>
#endif

#if defined(__cplusplus)
extern "C" {
#endif
extern void sleep(int);
#if defined(__cplusplus)
}
#endif

#undef __FUNC__  
#define __FUNC__ "PetscSleep"
/*@
   PetscSleep - Sleeps some number of seconds.

   Input Parameters:
.  s - number of seconds to sleep

.keywords: sleep, wait
@*/
void PetscSleep(int s)
{
  PetscFunctionBegin;
  if (s < 0) getc(stdin);
#if defined (PARCH_nt)
  else       _sleep(s*1000);
#else
  else       sleep(s);
#endif
  PetscFunctionReturn(0);
}

