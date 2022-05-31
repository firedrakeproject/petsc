#if !defined(GARBAGECOLLECTOR_H)
#define GARBAGECOLLECTOR_H

#include <petsc/private/hashmapobj.h>
#include <petscsys.h>

PETSC_EXTERN PetscErrorCode PetscObjectDelayedDestroy(PetscObject*);
PETSC_EXTERN PetscErrorCode PetscGarbageCleanup(MPI_Comm);

PETSC_EXTERN PetscErrorCode PrintGarbage_Private(MPI_Comm);

#endif
