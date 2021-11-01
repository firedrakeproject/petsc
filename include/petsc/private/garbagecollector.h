#if !defined(GARBAGECOLLECTOR_H)
#define GARBAGECOLLECTOR_H

#include "petsc/private/hashmapobj.h"
#include <petscsys.h>

PETSC_EXTERN PetscMPIInt GARBAGE;
PETSC_EXTERN PetscMPIInt INTRA;
PETSC_EXTERN PetscMPIInt INTER;

PETSC_EXTERN void sorted_intersect(int *seta, int *lena, int *setb, int lenb);
PETSC_EXTERN PetscErrorCode DelayedObjectDestroy(PetscObject *obj);
PETSC_EXTERN PetscErrorCode PetscGarbageCleanup(MPI_Comm comm, PetscInt blocksize);

#endif
