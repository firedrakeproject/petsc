# --------------------------------------------------------------------

class DeviceType(object):
    INVALID = PETSC_DEVICE_INVALID
    CUDA    = PETSC_DEVICE_CUDA
    HIP     = PETSC_DEVICE_HIP
    SYCL    = PETSC_DEVICE_SYCL
    MAX     = PETSC_DEVICE_MAX

# --------------------------------------------------------------------

cdef class Device(Object):
    Type = DeviceType

    @staticmethod
    def initialize_device(device_type):
        CHKERR ( PetscDeviceInitialize(device_type) )

    @staticmethod
    def is_device_initialized(device_type):
        return bool(PetscDeviceInitialized(device_type))

del DeviceType

# --------------------------------------------------------------------
