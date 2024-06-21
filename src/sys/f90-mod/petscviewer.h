!
!  Used by petscsysmod.F90 to create Fortran module file
!
#include "petsc/finclude/petscviewer.h"

      type, extends(tPetscObject) :: tPetscViewer
      end type tPetscViewer
      PetscViewer, parameter :: PETSC_NULL_VIEWER = tPetscViewer(0)
#if defined(_WIN32) && defined(PETSC_USE_SHARED_LIBRARIES)
!DEC$ ATTRIBUTES DLLEXPORT::PETSC_NULL_VIEWER
#endif
!
!     The numbers used below should match those in
!     petsc/private/fortranimpl.h
!
      PetscViewer, parameter :: PETSC_VIEWER_STDOUT_SELF  = tPetscViewer(9)
      PetscViewer, parameter :: PETSC_VIEWER_DRAW_WORLD   = tPetscViewer(4)
      PetscViewer, parameter :: PETSC_VIEWER_DRAW_SELF    = tPetscViewer(5)
      PetscViewer, parameter :: PETSC_VIEWER_SOCKET_WORLD = tPetscViewer(6)
      PetscViewer, parameter :: PETSC_VIEWER_SOCKET_SELF  = tPetscViewer(7)
      PetscViewer, parameter :: PETSC_VIEWER_STDOUT_WORLD = tPetscViewer(8)
      PetscViewer, parameter :: PETSC_VIEWER_STDERR_WORLD = tPetscViewer(10)
      PetscViewer, parameter :: PETSC_VIEWER_STDERR_SELF  = tPetscViewer(11)
      PetscViewer, parameter :: PETSC_VIEWER_BINARY_WORLD = tPetscViewer(12)
      PetscViewer, parameter :: PETSC_VIEWER_BINARY_SELF  = tPetscViewer(13)
      PetscViewer, parameter :: PETSC_VIEWER_MATLAB_WORLD = tPetscViewer(14)
      PetscViewer, parameter :: PETSC_VIEWER_MATLAB_SELF  = tPetscViewer(15)

      PetscViewer PETSC_VIEWER_STDOUT_
      PetscViewer PETSC_VIEWER_DRAW_
      external PETSC_VIEWER_STDOUT_
      external PETSC_VIEWER_DRAW_
      external PetscViewerAndFormatDestroy
!
!  Flags for binary I/O
!
      PetscEnum, parameter :: FILE_MODE_READ = 0
      PetscEnum, parameter :: FILE_MODE_WRITE = 1
      PetscEnum, parameter :: FILE_MODE_APPEND = 2
      PetscEnum, parameter :: FILE_MODE_UPDATE = 3
      PetscEnum, parameter :: FILE_MODE_APPEND_UPDATE = 4
!
!  PetscViewer formats
!
      PetscEnum, parameter :: PETSC_VIEWER_DEFAULT = 0
      PetscEnum, parameter :: PETSC_VIEWER_ASCII_MATLAB = 1
      PetscEnum, parameter :: PETSC_VIEWER_ASCII_MATHEMATICA = 2
      PetscEnum, parameter :: PETSC_VIEWER_ASCII_IMPL = 3
      PetscEnum, parameter :: PETSC_VIEWER_ASCII_INFO = 4
      PetscEnum, parameter :: PETSC_VIEWER_ASCII_INFO_DETAIL = 5
      PetscEnum, parameter :: PETSC_VIEWER_ASCII_COMMON = 6
      PetscEnum, parameter :: PETSC_VIEWER_ASCII_SYMMODU = 7
      PetscEnum, parameter :: PETSC_VIEWER_ASCII_INDEX = 8
      PetscEnum, parameter :: PETSC_VIEWER_ASCII_DENSE = 9
      PetscEnum, parameter :: PETSC_VIEWER_ASCII_MATRIXMARKET = 10
      PetscEnum, parameter :: PETSC_VIEWER_ASCII_VTK = 11
      PetscEnum, parameter :: PETSC_VIEWER_ASCII_VTK_CELL = 12
      PetscEnum, parameter :: PETSC_VIEWER_ASCII_VTK_COORDS = 13
      PetscEnum, parameter :: PETSC_VIEWER_ASCII_PCICE = 14
      PetscEnum, parameter :: PETSC_VIEWER_ASCII_PYTHON = 15
      PetscEnum, parameter :: PETSC_VIEWER_ASCII_FACTOR_INFO = 16
      PetscEnum, parameter :: PETSC_VIEWER_ASCII_LATEX = 17
      PetscEnum, parameter :: PETSC_VIEWER_ASCII_XML = 18
      PetscEnum, parameter :: PETSC_VIEWER_ASCII_GLVIS = 19
      PetscEnum, parameter :: PETSC_VIEWER_ASCII_CSV = 20
      PetscEnum, parameter :: PETSC_VIEWER_DRAW_BASIC = 21
      PetscEnum, parameter :: PETSC_VIEWER_DRAW_LG = 22
      PetscEnum, parameter :: PETSC_VIEWER_DRAW_LG_XRANGE = 23
      PetscEnum, parameter :: PETSC_VIEWER_DRAW_CONTOUR = 24
      PetscEnum, parameter :: PETSC_VIEWER_DRAW_PORTS = 25
      PetscEnum, parameter :: PETSC_VIEWER_VTK_VTS = 26
      PetscEnum, parameter :: PETSC_VIEWER_VTK_VTR = 27
      PetscEnum, parameter :: PETSC_VIEWER_VTK_VTU = 28
      PetscEnum, parameter :: PETSC_VIEWER_BINARY_MATLAB = 29
      PetscEnum, parameter :: PETSC_VIEWER_NATIVE = 30
      PetscEnum, parameter :: PETSC_VIEWER_HDF5_PETSC = 31
      PetscEnum, parameter :: PETSC_VIEWER_HDF5_VIZ = 32
      PetscEnum, parameter :: PETSC_VIEWER_HDF5_XDMF = 33
      PetscEnum, parameter :: PETSC_VIEWER_HDF5_MAT = 34
      PetscEnum, parameter :: PETSC_VIEWER_NOFORMAT = 35
      PetscEnum, parameter :: PETSC_VIEWER_LOAD_BALANCE = 36
      PetscEnum, parameter :: PETSC_VIEWER_LOAD_ALL = 37

#if defined(_WIN32) && defined(PETSC_USE_SHARED_LIBRARIES)
!DEC$ ATTRIBUTES DLLEXPORT::PETSC_VIEWER_STDOUT_SELF
!DEC$ ATTRIBUTES DLLEXPORT::PETSC_VIEWER_DRAW_WORLD
!DEC$ ATTRIBUTES DLLEXPORT::PETSC_VIEWER_DRAW_SELF
!DEC$ ATTRIBUTES DLLEXPORT::PETSC_VIEWER_SOCKET_WORLD
!DEC$ ATTRIBUTES DLLEXPORT::PETSC_VIEWER_SOCKET_SELF
!DEC$ ATTRIBUTES DLLEXPORT::PETSC_VIEWER_STDOUT_WORLD
!DEC$ ATTRIBUTES DLLEXPORT::PETSC_VIEWER_STDERR_WORLD
!DEC$ ATTRIBUTES DLLEXPORT::PETSC_VIEWER_STDERR_SELF
!DEC$ ATTRIBUTES DLLEXPORT::PETSC_VIEWER_BINARY_WORLD
!DEC$ ATTRIBUTES DLLEXPORT::PETSC_VIEWER_BINARY_SELF
!DEC$ ATTRIBUTES DLLEXPORT::PETSC_VIEWER_MATLAB_WORLD
!DEC$ ATTRIBUTES DLLEXPORT::PETSC_VIEWER_MATLAB_SELF
!DEC$ ATTRIBUTES DLLEXPORT::FILE_MODE_READ
!DEC$ ATTRIBUTES DLLEXPORT::FILE_MODE_WRITE
!DEC$ ATTRIBUTES DLLEXPORT::FILE_MODE_APPEND
!DEC$ ATTRIBUTES DLLEXPORT::FILE_MODE_UPDATE
!DEC$ ATTRIBUTES DLLEXPORT::FILE_MODE_APPEND_UPDATE
!DEC$ ATTRIBUTES DLLEXPORT::PETSC_VIEWER_DEFAULT
!DEC$ ATTRIBUTES DLLEXPORT::PETSC_VIEWER_ASCII_MATLAB
!DEC$ ATTRIBUTES DLLEXPORT::PETSC_VIEWER_ASCII_MATHEMATICA
!DEC$ ATTRIBUTES DLLEXPORT::PETSC_VIEWER_ASCII_IMPL
!DEC$ ATTRIBUTES DLLEXPORT::PETSC_VIEWER_ASCII_INFO
!DEC$ ATTRIBUTES DLLEXPORT::PETSC_VIEWER_ASCII_INFO_DETAIL
!DEC$ ATTRIBUTES DLLEXPORT::PETSC_VIEWER_ASCII_COMMON
!DEC$ ATTRIBUTES DLLEXPORT::PETSC_VIEWER_ASCII_SYMMODU
!DEC$ ATTRIBUTES DLLEXPORT::PETSC_VIEWER_ASCII_INDEX
!DEC$ ATTRIBUTES DLLEXPORT::PETSC_VIEWER_ASCII_DENSE
!DEC$ ATTRIBUTES DLLEXPORT::PETSC_VIEWER_ASCII_MATRIXMARKET
!DEC$ ATTRIBUTES DLLEXPORT::PETSC_VIEWER_ASCII_VTK
!DEC$ ATTRIBUTES DLLEXPORT::PETSC_VIEWER_ASCII_VTK_CELL
!DEC$ ATTRIBUTES DLLEXPORT::PETSC_VIEWER_ASCII_VTK_COORDS
!DEC$ ATTRIBUTES DLLEXPORT::PETSC_VIEWER_ASCII_PCICE
!DEC$ ATTRIBUTES DLLEXPORT::PETSC_VIEWER_ASCII_PYTHON
!DEC$ ATTRIBUTES DLLEXPORT::PETSC_VIEWER_ASCII_FACTOR_INFO
!DEC$ ATTRIBUTES DLLEXPORT::PETSC_VIEWER_ASCII_LATEX
!DEC$ ATTRIBUTES DLLEXPORT::PETSC_VIEWER_ASCII_XML
!DEC$ ATTRIBUTES DLLEXPORT::PETSC_VIEWER_ASCII_GLVIS
!DEC$ ATTRIBUTES DLLEXPORT::PETSC_VIEWER_DRAW_BASIC
!DEC$ ATTRIBUTES DLLEXPORT::PETSC_VIEWER_DRAW_LG
!DEC$ ATTRIBUTES DLLEXPORT::PETSC_VIEWER_DRAW_CONTOUR
!DEC$ ATTRIBUTES DLLEXPORT::PETSC_VIEWER_DRAW_PORTS
!DEC$ ATTRIBUTES DLLEXPORT::PETSC_VIEWER_VTK_VTS
!DEC$ ATTRIBUTES DLLEXPORT::PETSC_VIEWER_VTK_VTR
!DEC$ ATTRIBUTES DLLEXPORT::PETSC_VIEWER_VTK_VTU
!DEC$ ATTRIBUTES DLLEXPORT::PETSC_VIEWER_BINARY_MATLAB
!DEC$ ATTRIBUTES DLLEXPORT::PETSC_VIEWER_NATIVE
!DEC$ ATTRIBUTES DLLEXPORT::PETSC_VIEWER_HDF5_VIZ
!DEC$ ATTRIBUTES DLLEXPORT::PETSC_VIEWER_NOFORMAT
!DEC$ ATTRIBUTES DLLEXPORT::PETSC_VIEWER_ALL
#endif
