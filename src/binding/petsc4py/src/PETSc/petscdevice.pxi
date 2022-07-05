# --------------------------------------------------------------------

cdef extern from * nogil:

    ctypedef enum PetscDeviceType:
        PETSC_DEVICE_INVALID
        PETSC_DEVICE_CUDA
        PETSC_DEVICE_HIP
        PETSC_DEVICE_SYCL
        PETSC_DEVICE_MAX

    int PetscDeviceInitialize(PetscDeviceType)
    PetscBool PetscDeviceInitialized(PetscDeviceType)

# --------------------------------------------------------------------
