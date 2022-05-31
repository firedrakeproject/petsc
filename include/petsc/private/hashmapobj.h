#if !defined(PETSC_HASHMAPP_H)
#define PETSC_HASHMAPP_H

#include <petsc/private/hashmap.h>

/*
 * Hash map from PetscInt --> PetscObject*
 * */
PETSC_HASH_MAP(HMapObj,PetscCount,PetscObject*,PetscHashInt,PetscHashEqual,NULL)

#endif /* PETSC_HASHMAPP_H */
