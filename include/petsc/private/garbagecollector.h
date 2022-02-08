#if !defined(GARBAGECOLLECTOR_H)
#define GARBAGECOLLECTOR_H

#include <petsc/private/hashmapobj.h>
#include <petscsys.h>

PETSC_EXTERN PetscErrorCode PetscObjectDelayedDestroy(PetscObject*);
PETSC_EXTERN PetscErrorCode PetscGarbageCleanup(MPI_Comm,PetscInt);
PETSC_EXTERN PetscErrorCode PetscGarbageRecursiveCleanup(MPI_Comm,PetscInt);

#endif
