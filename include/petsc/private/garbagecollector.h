#if !defined(GARBAGECOLLECTOR_H)
#define GARBAGECOLLECTOR_H

#include <petsc/private/hashmapobj.h>
#include <petscsys.h>

PETSC_EXTERN PetscErrorCode PetscObjectDelayedDestroy(PetscObject*);
PETSC_EXTERN void PetscGarbageKeySortedIntersect(void*,void*,PetscMPIInt*,MPI_Datatype*);
PETSC_EXTERN PetscErrorCode PetscGarbageCleanup(MPI_Comm);
PETSC_EXTERN PetscErrorCode PetscGarbagePrint_Private(MPI_Comm);

#endif
