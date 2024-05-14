#include <../src/sys/classes/viewer/impls/ascii/asciiimpl.h> /*I     "petscviewer.h"   I*/

/*
    The variable Petsc_Viewer_Stdout_keyval is used to indicate an MPI attribute that
  is attached to a communicator, in this case the attribute is a PetscViewer.
*/
PetscMPIInt Petsc_Viewer_Stdout_keyval = MPI_KEYVAL_INVALID;

/*@C
   PETSC_VIEWER_STDOUT_ - Creates a `PETSCVIEWERASCII` `PetscViewer` shared by all MPI processes
                    in a communicator.

   Collective

   Input Parameter:
.  comm - the MPI communicator to share the `PetscViewer`

   Level: beginner

   Notes:
   This object is destroyed in `PetscFinalize()`, `PetscViewerDestroy()` should never be called on it

   Unlike almost all other PETSc routines, this does not return
   an error code. Usually used in the form
$      XXXView(XXX object, PETSC_VIEWER_STDOUT_(comm));

.seealso: [](sec_viewers), `PETSC_VIEWER_DRAW_()`, `PetscViewerASCIIOpen()`, `PETSC_VIEWER_STDERR_`, `PETSC_VIEWER_STDOUT_WORLD`,
          `PETSC_VIEWER_STDOUT_SELF`
@*/
PetscViewer PETSC_VIEWER_STDOUT_(MPI_Comm comm)
{
  PetscErrorCode ierr;
  PetscViewer    viewer;

  PetscFunctionBegin;
  ierr = PetscViewerASCIIGetStdout(comm, &viewer);
  if (ierr) {
    ierr = PetscError(PETSC_COMM_SELF, __LINE__, "PETSC_VIEWER_STDOUT_", __FILE__, PETSC_ERR_PLIB, PETSC_ERROR_INITIAL, " ");
    PetscFunctionReturn(NULL);
  }
  PetscFunctionReturn(viewer);
}

/*
    The variable Petsc_Viewer_Stderr_keyval is used to indicate an MPI attribute that
  is attached to a communicator, in this case the attribute is a PetscViewer.
*/
PetscMPIInt Petsc_Viewer_Stderr_keyval = MPI_KEYVAL_INVALID;

/*@
  PetscViewerASCIIGetStderr - Creates a `PETSCVIEWERASCII` `PetscViewer` shared by all MPI processes
  in a communicator. Error returning version of `PETSC_VIEWER_STDERR_()`

  Collective

  Input Parameter:
. comm - the MPI communicator to share the `PetscViewer`

  Output Parameter:
. viewer - the viewer

  Level: beginner

  Note:
  This object is destroyed in `PetscFinalize()`, `PetscViewerDestroy()` should never be called on it

  Developer Note:
  This should be used in all PETSc source code instead of `PETSC_VIEWER_STDERR_()` since it allows error checking

.seealso: [](sec_viewers), `PETSC_VIEWER_DRAW_()`, `PetscViewerASCIIOpen()`, `PETSC_VIEWER_STDERR_`, `PETSC_VIEWER_STDERR_WORLD`,
          `PETSC_VIEWER_STDERR_SELF`
@*/
PetscErrorCode PetscViewerASCIIGetStderr(MPI_Comm comm, PetscViewer *viewer)
{
  PetscBool flg;
  MPI_Comm  ncomm;

  PetscFunctionBegin;
  PetscCall(PetscSpinlockLock(&PetscViewerASCIISpinLockStderr));
  PetscCall(PetscCommDuplicate(comm, &ncomm, NULL));
  if (Petsc_Viewer_Stderr_keyval == MPI_KEYVAL_INVALID) PetscCallMPI(MPI_Comm_create_keyval(MPI_COMM_NULL_COPY_FN, MPI_COMM_NULL_DELETE_FN, &Petsc_Viewer_Stderr_keyval, NULL));
  PetscCallMPI(MPI_Comm_get_attr(ncomm, Petsc_Viewer_Stderr_keyval, (void **)viewer, (PetscMPIInt *)&flg));
  if (!flg) { /* PetscViewer not yet created */
    PetscCall(PetscViewerASCIIOpen(ncomm, "stderr", viewer));
    ((PetscObject)*viewer)->persistent = PETSC_TRUE;
    PetscCall(PetscObjectRegisterDestroy((PetscObject)*viewer));
    PetscCallMPI(MPI_Comm_set_attr(ncomm, Petsc_Viewer_Stderr_keyval, (void *)*viewer));
  }
  PetscCall(PetscCommDestroy(&ncomm));
  PetscCall(PetscSpinlockUnlock(&PetscViewerASCIISpinLockStderr));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
   PETSC_VIEWER_STDERR_ - Creates a `PETSCVIEWERASCII` `PetscViewer` shared by all MPI processes
                    in a communicator.

   Collective

   Input Parameter:
.  comm - the MPI communicator to share the `PetscViewer`

   Level: beginner

   Notes:
   This object is destroyed in `PetscFinalize()`, `PetscViewerDestroy()` should never be called on it

   Unlike almost all other PETSc routines, this does not return
   an error code. Usually used in the form
$      XXXView(XXX object, PETSC_VIEWER_STDERR_(comm));

   `PetscViewerASCIIGetStderr()` is preferred  since it allows error checking

.seealso: [](sec_viewers), `PETSC_VIEWER_DRAW_`, `PetscViewerASCIIOpen()`, `PETSC_VIEWER_STDOUT_`, `PETSC_VIEWER_STDOUT_WORLD`,
          `PETSC_VIEWER_STDOUT_SELF`, `PETSC_VIEWER_STDERR_WORLD`, `PETSC_VIEWER_STDERR_SELF`
@*/
PetscViewer PETSC_VIEWER_STDERR_(MPI_Comm comm)
{
  PetscErrorCode ierr;
  PetscViewer    viewer;

  PetscFunctionBegin;
  ierr = PetscViewerASCIIGetStderr(comm, &viewer);
  if (ierr) {
    ierr = PetscError(PETSC_COMM_SELF, __LINE__, "PETSC_VIEWER_STDERR_", __FILE__, PETSC_ERR_PLIB, PETSC_ERROR_INITIAL, " ");
    PetscFunctionReturn(NULL);
  }
  PetscFunctionReturn(viewer);
}

PetscMPIInt Petsc_Viewer_keyval = MPI_KEYVAL_INVALID;
/*
   Called with MPI_Comm_free() is called on a communicator that has a viewer as an attribute. The viewer is not actually destroyed
   because that is managed by PetscObjectDestroyRegisterAll(). PetscViewerASCIIGetStdout() registers the viewer with PetscObjectDestroyRegister() to be destroyed when PetscFinalize() is called.

  This is called by MPI, not by users.

*/
PetscMPIInt MPIAPI Petsc_DelViewer(MPI_Comm comm, PetscMPIInt keyval, void *attr_val, void *extra_state)
{
  PetscFunctionBegin;
  (void)keyval;
  (void)attr_val;
  (void)extra_state;
  PetscCallMPI(PetscInfo(NULL, "Removing viewer data attribute in an MPI_Comm %" PETSC_INTPTR_T_FMT "\n", (PETSC_INTPTR_T)comm));
  PetscFunctionReturn(MPI_SUCCESS);
}

/*@
  PetscViewerASCIIOpen - Opens an ASCII file for writing as a `PETSCVIEWERASCII` `PetscViewer`.

  Collective

  Input Parameters:
+ comm - the communicator
- name - the file name

  Output Parameter:
. lab - the `PetscViewer` to use with the specified file

  Level: beginner

  Notes:
  To open a ASCII file as a viewer for reading one must use the sequence
.vb
   PetscViewerCreate(comm,&lab);
   PetscViewerSetType(lab,PETSCVIEWERASCII);
   PetscViewerFileSetMode(lab,FILE_MODE_READ);
   PetscViewerFileSetName(lab,name);
.ve

  This `PetscViewer` can be destroyed with `PetscViewerDestroy()`.

  The MPI communicator used here must match that used by the object one is viewing. For example if the
  Mat was created with a `PETSC_COMM_WORLD`, then the Viewer must be created with `PETSC_COMM_WORLD`

  As shown below, `PetscViewerASCIIOpen()` is useful in conjunction with
  `MatView()` and `VecView()`
.vb
     PetscViewerASCIIOpen(PETSC_COMM_WORLD,"mat.output",&viewer);
     MatView(matrix,viewer);
.ve

.seealso: [](sec_viewers), `MatView()`, `VecView()`, `PetscViewerDestroy()`, `PetscViewerBinaryOpen()`, `PetscViewerASCIIRead()`, `PETSCVIEWERASCII`
          `PetscViewerASCIIGetPointer()`, `PetscViewerPushFormat()`, `PETSC_VIEWER_STDOUT_`, `PETSC_VIEWER_STDERR_`,
          `PETSC_VIEWER_STDOUT_WORLD`, `PETSC_VIEWER_STDOUT_SELF`,
@*/
PetscErrorCode PetscViewerASCIIOpen(MPI_Comm comm, const char name[], PetscViewer *lab)
{
  PetscViewerLink *vlink, *nv;
  PetscBool        flg, eq;
  size_t           len;

  PetscFunctionBegin;
  PetscCall(PetscStrlen(name, &len));
  if (!len) {
    PetscCall(PetscViewerASCIIGetStdout(comm, lab));
    PetscCall(PetscObjectReference((PetscObject)*lab));
    PetscFunctionReturn(PETSC_SUCCESS);
  }
  PetscCall(PetscSpinlockLock(&PetscViewerASCIISpinLockOpen));
  if (Petsc_Viewer_keyval == MPI_KEYVAL_INVALID) PetscCallMPI(MPI_Comm_create_keyval(MPI_COMM_NULL_COPY_FN, Petsc_DelViewer, &Petsc_Viewer_keyval, (void *)0));
  /*
       It would be better to move this code to PetscFileSetName() but since it must return a preexiting communicator
     we cannot do that, since PetscFileSetName() takes a communicator that already exists.

      Plus if the original communicator that created the file has since been close this will not detect the old
      communictor and hence will overwrite the old data. It may be better to simply remove all this code
  */
  /* make sure communicator is a PETSc communicator */
  PetscCall(PetscCommDuplicate(comm, &comm, NULL));
  /* has file already been opened into a viewer */
  PetscCallMPI(MPI_Comm_get_attr(comm, Petsc_Viewer_keyval, (void **)&vlink, (PetscMPIInt *)&flg));
  if (flg) {
    while (vlink) {
      PetscCall(PetscStrcmp(name, ((PetscViewer_ASCII *)vlink->viewer->data)->filename, &eq));
      if (eq) {
        PetscCall(PetscObjectReference((PetscObject)vlink->viewer));
        *lab = vlink->viewer;
        PetscCall(PetscCommDestroy(&comm));
        PetscCall(PetscSpinlockUnlock(&PetscViewerASCIISpinLockOpen));
        PetscFunctionReturn(PETSC_SUCCESS);
      }
      vlink = vlink->next;
    }
  }
  PetscCall(PetscViewerCreate(comm, lab));
  PetscCall(PetscViewerSetType(*lab, PETSCVIEWERASCII));
  if (name) PetscCall(PetscViewerFileSetName(*lab, name));
  /* save viewer into communicator if needed later */
  PetscCall(PetscNew(&nv));
  nv->viewer = *lab;
  if (!flg) {
    PetscCallMPI(MPI_Comm_set_attr(comm, Petsc_Viewer_keyval, nv));
  } else {
    PetscCallMPI(MPI_Comm_get_attr(comm, Petsc_Viewer_keyval, (void **)&vlink, (PetscMPIInt *)&flg));
    if (vlink) {
      while (vlink->next) vlink = vlink->next;
      vlink->next = nv;
    } else {
      PetscCallMPI(MPI_Comm_set_attr(comm, Petsc_Viewer_keyval, nv));
    }
  }
  PetscCall(PetscCommDestroy(&comm));
  PetscCall(PetscSpinlockUnlock(&PetscViewerASCIISpinLockOpen));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  PetscViewerASCIIOpenWithFILE - Given an open file creates an `PETSCVIEWERASCII` viewer that prints to it.

  Collective

  Input Parameters:
+ comm - the communicator
- fd   - the `FILE` pointer

  Output Parameter:
. lab - the `PetscViewer` to use with the specified file

  Level: beginner

  Notes:
  This `PetscViewer` can be destroyed with `PetscViewerDestroy()`, but the fd will NOT be closed.

  If a multiprocessor communicator is used (such as `PETSC_COMM_WORLD`),
  then only the first processor in the group uses the file.  All other
  processors send their data to the first processor to print.

  Fortran Notes:
  Use `PetscViewerASCIIOpenWithFileUnit()`

.seealso: [](sec_viewers), `MatView()`, `VecView()`, `PetscViewerDestroy()`, `PetscViewerBinaryOpen()`, `PetscViewerASCIIOpenWithFileUnit()`,
          `PetscViewerASCIIGetPointer()`, `PetscViewerPushFormat()`, `PETSC_VIEWER_STDOUT_`, `PETSC_VIEWER_STDERR_`,
          `PETSC_VIEWER_STDOUT_WORLD`, `PETSC_VIEWER_STDOUT_SELF`, `PetscViewerASCIIOpen()`, `PetscViewerASCIISetFILE()`, `PETSCVIEWERASCII`
@*/
PetscErrorCode PetscViewerASCIIOpenWithFILE(MPI_Comm comm, FILE *fd, PetscViewer *lab)
{
  PetscFunctionBegin;
  PetscCall(PetscViewerCreate(comm, lab));
  PetscCall(PetscViewerSetType(*lab, PETSCVIEWERASCII));
  PetscCall(PetscViewerASCIISetFILE(*lab, fd));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  PetscViewerASCIISetFILE - Given an open file sets the `PETSCVIEWERASCII` viewer to use the file for output

  Not Collective

  Input Parameters:
+ viewer - the `PetscViewer` to use with the specified file
- fd     - the `FILE` pointer

  Level: beginner

  Notes:
  This `PetscViewer` can be destroyed with `PetscViewerDestroy()`, but the `fd` will NOT be closed.

  If a multiprocessor communicator is used (such as `PETSC_COMM_WORLD`),
  then only the first processor in the group uses the file.  All other
  processors send their data to the first processor to print.

  Fortran Notes:
  Use `PetscViewerASCIISetFileUnit()`

.seealso: `MatView()`, `VecView()`, `PetscViewerDestroy()`, `PetscViewerBinaryOpen()`, `PetscViewerASCIISetFileUnit()`,
          `PetscViewerASCIIGetPointer()`, `PetscViewerPushFormat()`, `PETSC_VIEWER_STDOUT_`, `PETSC_VIEWER_STDERR_`,
          `PETSC_VIEWER_STDOUT_WORLD`, `PETSC_VIEWER_STDOUT_SELF`, `PetscViewerASCIIOpen()`, `PetscViewerASCIIOpenWithFILE()`, `PETSCVIEWERASCII`
@*/
PetscErrorCode PetscViewerASCIISetFILE(PetscViewer viewer, FILE *fd)
{
  PetscViewer_ASCII *vascii = (PetscViewer_ASCII *)viewer->data;

  PetscFunctionBegin;
  vascii->fd        = fd;
  vascii->closefile = PETSC_FALSE;
  PetscFunctionReturn(PETSC_SUCCESS);
}
