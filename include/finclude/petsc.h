!
!  $Id: petsc.h,v 1.98 2001/08/10 16:50:53 balay Exp balay $;
!
!  Base include file for Fortran use of the PETSc package.
!
#include "petscconf.h"

#if !defined(PETSC_AVOID_MPIF_H) && !defined(PETSC_AVOID_DECLARATIONS)
#include "mpif.h"
#endif

#include "finclude/petscdef.h"

#if !defined (PETSC_AVOID_DECLARATIONS)
! ------------------------------------------------------------------------
!     Non Common block Stuff declared first
!    
!     Flags
!
      integer   PETSC_TRUE,PETSC_FALSE
      integer   PETSC_YES,PETSC_NO
      parameter (PETSC_TRUE = 1,PETSC_FALSE = 0)
      parameter (PETSC_YES=1, PETSC_NO=0)

      integer   PETSC_DECIDE,PETSC_DETERMINE
      parameter (PETSC_DECIDE=-1,PETSC_DETERMINE=-1)

      integer   PETSC_DEFAULT_INTEGER
      parameter (PETSC_DEFAULT_INTEGER = -2)

      PetscFortranDouble PETSC_DEFAULT_DOUBLE_PRECISION
      parameter (PETSC_DEFAULT_DOUBLE_PRECISION=-2.0d0)

      integer   PETSC_FP_TRAP_OFF,PETSC_FP_TRAP_ON
      parameter (PETSC_FP_TRAP_OFF = 0,PETSC_FP_TRAP_ON = 1) 



!
!     Default PetscViewers.
!
      PetscFortranAddr PETSC_VIEWER_DRAW_WORLD
      PetscFortranAddr PETSC_VIEWER_DRAW_SELF
      PetscFortranAddr PETSC_VIEWER_SOCKET_WORLD
      PetscFortranAddr PETSC_VIEWER_SOCKET_SELF
      PetscFortranAddr PETSC_VIEWER_STDOUT_WORLD
      PetscFortranAddr PETSC_VIEWER_STDOUT_SELF
      PetscFortranAddr PETSC_VIEWER_STDERR_WORLD
      PetscFortranAddr PETSC_VIEWER_STDERR_SELF
      PetscFortranAddr PETSC_VIEWER_BINARY_WORLD
      PetscFortranAddr PETSC_VIEWER_BINARY_SELF

!
!     The numbers used below should match those in 
!     src/fortran/custom/zpetsc.h
!
      parameter (PETSC_VIEWER_DRAW_WORLD   = -4) 
      parameter (PETSC_VIEWER_DRAW_SELF    = -5)
      parameter (PETSC_VIEWER_SOCKET_WORLD = -6)
      parameter (PETSC_VIEWER_SOCKET_SELF  = -7)
      parameter (PETSC_VIEWER_STDOUT_WORLD = -8)
      parameter (PETSC_VIEWER_STDOUT_SELF  = -9)
      parameter (PETSC_VIEWER_STDERR_WORLD = -10)
      parameter (PETSC_VIEWER_STDERR_SELF  = -11)
      parameter (PETSC_VIEWER_BINARY_WORLD = -12)
      parameter (PETSC_VIEWER_BINARY_SELF  = -13)
!
!     PETSc DataTypes
!
      integer PETSC_INT,PETSC_DOUBLE,PETSC_COMPLEX
      integer PETSC_LONG,PETSC_SHORT,PETSC_FLOAT
      integer PETSC_CHAR,PETSC_LOGICAL

      parameter (PETSC_INT=0,PETSC_DOUBLE=1,PETSC_COMPLEX=2)
      parameter (PETSC_LONG=3,PETSC_SHORT=4,PETSC_FLOAT=5)
      parameter (PETSC_CHAR=6,PETSC_LOGICAL=7)
!
! ------------------------------------------------------------------------
!     PETSc mathematics include file. Defines certain basic mathematical 
!    constants and functions for working with single and double precision
!    floating point numbers as well as complex and integers.
!
!     Representation of complex i
!
      PetscFortranComplex PETSC_i
      parameter (PETSC_i = (0.0d0,1.0d0))
!
!     Basic constants
! 
      PetscFortranDouble PETSC_PI,PETSC_DEGREES_TO_RADIANS
      PetscFortranDouble PETSC_MAX,PETSC_MIN

      parameter (PETSC_PI = 3.14159265358979323846264d0)
      parameter (PETSC_DEGREES_TO_RADIANS = 0.01745329251994d0)
      parameter (PETSC_MAX = 1.d300,PETSC_MIN = -1.d300)

      PetscFortranDouble PETSC_MACHINE_EPSILON
      PetscFortranDouble PETSC_SQRT_MACHINE_EPSILON
      PetscFortranDouble PETSC_SMALL

#if defined(PETSC_USE_SINGLE)
      parameter (PETSC_MACHINE_EPSILON = 1.e-7)
      parameter (PETSC_SQRT_MACHINE_EPSILON = 3.e-4)
      parameter (PETSC_SMALL = 1.e-5)
#else
      parameter (PETSC_MACHINE_EPSILON = 1.d-14)
      parameter (PETSC_SQRT_MACHINE_EPSILON = 1.d-7)
      parameter (PETSC_SMALL = 1.d-10)
#endif
!
! ----------------------------------------------------------------------------
!    BEGIN COMMON-BLOCK VARIABLES
!
!
!     PETSc world communicator
!
      MPI_Comm PETSC_COMM_WORLD,PETSC_COMM_SELF
!
!     Fortran Null
!
      character*(80)      PETSC_NULL_CHARACTER
      PetscFortranInt     PETSC_NULL_INTEGER
      PetscFortranDouble  PETSC_NULL_DOUBLE
!
!      A PETSC_NULL_FUNCTION pointer
!
      external PETSC_NULL_FUNCTION
      PetscScalar   PETSC_NULL_SCALAR
      PetscReal     PETSC_NULL_REAL
!
!     Common Block to store some of the PETSc constants.
!     which can be set - only at runtime.
!
!
!     A string should be in a different common block
!  
      common /petscfortran1/ PETSC_NULL_CHARACTER
      common /petscfortran2/ PETSC_NULL_INTEGER
      common /petscfortran3/ PETSC_NULL_SCALAR
      common /petscfortran4/ PETSC_NULL_DOUBLE
      common /petscfortran5/ PETSC_NULL_REAL
      common /petscfortran6/ PETSC_COMM_WORLD,PETSC_COMM_SELF

!    END COMMON-BLOCK VARIABLES
! ----------------------------------------------------------------------------

#endif
