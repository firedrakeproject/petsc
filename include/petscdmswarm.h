#if !defined(__PETSCDMSWARM_H)
#define __PETSCDMSWARM_H

#include <petscdm.h>

PETSC_EXTERN PetscErrorCode DMSwarmCreateGlobalVectorFromField(DM dm,const char fieldname[],Vec *vec);
PETSC_EXTERN PetscErrorCode DMSwarmDestroyGlobalVectorFromField(DM dm,const char fieldname[],Vec *vec);

PETSC_EXTERN PetscErrorCode DMSwarmInitializeFieldRegister(DM dm);
PETSC_EXTERN PetscErrorCode DMSwarmFinalizeFieldRegister(DM dm);
PETSC_EXTERN PetscErrorCode DMSwarmSetLocalSizes(DM dm,PetscInt nlocal,PetscInt buffer);
PETSC_EXTERN PetscErrorCode DMSwarmRegisterPetscDatatypeField(DM dm,const char fieldname[],PetscInt blocksize,PetscDataType type);
PETSC_EXTERN PetscErrorCode DMSwarmRegisterUserStructField(DM dm,const char fieldname[],size_t size);
PETSC_EXTERN PetscErrorCode DMSwarmRegisterUserDatatypeField(DM dm,const char fieldname[],size_t size);
PETSC_EXTERN PetscErrorCode DMSwarmGetField(DM dm,const char fieldname[],PetscInt *blocksize,PetscDataType *type,void **data);
PETSC_EXTERN PetscErrorCode DMSwarmRestoreField(DM dm,const char fieldname[],PetscInt *blocksize,PetscDataType *type,void **data);

PETSC_EXTERN PetscErrorCode DMSwarmVectorDefineField(DM dm,const char fieldname[]);

#endif

