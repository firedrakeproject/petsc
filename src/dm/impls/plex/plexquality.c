#include <petsc/private/dmpleximpl.h>


static PetscErrorCode DMPlexGetAspectRatio_Triangle(DM dm, Vec *aspectRatio) {
  DM                 cdm;
  PetscErrorCode     ierr;
  PetscInt           dim, cStart, cEnd, c, i;
  PetscScalar       *ar;
  const PetscScalar *coords;
  Vec                coordinates;

  PetscFunctionBegin;
  PetscCall(DMGetCoordinateDM(dm, &cdm));
  PetscCall(DMGetDimension(dm, &dim));
  PetscCall(DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd));
  PetscCall(DMGetCoordinates(dm, &coordinates));
  PetscCall(VecGetArrayRead(coordinates, &coords));
  PetscCall(VecGetArrayWrite(*aspectRatio, &ar));
  for (c = cStart; c < cEnd; ++c) {
    PetscScalar       *car, l[3];
    const PetscScalar *x1, *x2;
    const PetscInt    *edges, *vertices;

    PetscCall(DMPlexPointLocalRef(dm, c, ar, &car));
    PetscCall(DMPlexGetCone(dm, c, &edges));
    for (i = 0; i < 3; ++i) {
      PetscCall(DMPlexGetCone(dm, edges[i], &vertices));
      PetscCall(DMPlexPointLocalRead(cdm, vertices[0], coords, &x1));
      PetscCall(DMPlexPointLocalRead(cdm, vertices[1], coords, &x2));
      l[i] = PetscPowReal(PetscSqr(x1[0]-x2[0]) + PetscSqr(x1[1]-x2[1]), 0.5);
    }
    car[0] = l[0]*l[1]*l[2]/((l[0]+l[1]-l[2])*(l[1]+l[2]-l[0])*(l[2]+l[0]-l[1]));
  }
  PetscCall(VecRestoreArrayWrite(*aspectRatio, &ar));
  PetscCall(VecRestoreArrayRead(coordinates, &coords));
  PetscFunctionReturn(0);
}


/*
  DMPlexGetAspectRatio - Compute the aspect ratio of each element

  Input Parameters:
. dm - the DM

  Output Parameter:
. ar - The aspect ratio field

  Level: beginner
*/
PetscErrorCode DMPlexGetAspectRatio(DM dm, Vec *ar) {
  DM             qdm;
  DMPolytopeType ct;
  MPI_Comm       comm;
  PetscBool      simplex;
  PetscErrorCode ierr;
  PetscInt       dim, cStart, cEnd, c, field = 0;
  PetscSection   sec;

  PetscFunctionBegin;

  /* Extract metadata from DM */
  PetscCall(DMGetDimension(dm, &dim));
  PetscCall(PetscObjectGetComm((PetscObject) dm, &comm));

  /* Setup cell-wise field */
  PetscCall(DMClone(dm, &qdm));
  PetscCall(PetscSectionCreate(comm, &sec));
  PetscCall(PetscSectionSetNumFields(sec, 1));
  PetscCall(PetscSectionSetFieldComponents(sec, field, 1));
  PetscCall(DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd));
  PetscCall(PetscSectionSetChart(sec, cStart, cEnd));
  for (c = cStart; c < cEnd; ++c) {
    PetscCall(PetscSectionSetDof(sec, c, 1));
    PetscCall(PetscSectionSetFieldDof(sec, c, field, 1));
  }
  PetscCall(PetscSectionSetUp(sec));
  PetscCall(DMSetLocalSection(qdm, sec));
  PetscCall(PetscSectionDestroy(&sec));
  PetscCall(DMCreateLocalVector(qdm, ar));

  /* Compute aspect ratio */
  PetscCall(DMPlexGetCellType(dm, cStart, &ct));
  simplex = DMPolytopeTypeGetNumVertices(ct) == DMPolytopeTypeGetDim(ct)+1 ? PETSC_TRUE : PETSC_FALSE;
  switch (dim) {
  case 2:
    if (simplex) PetscCall(DMPlexGetAspectRatio_Triangle(qdm, ar));
    else SETERRQ(comm, PETSC_ERR_ARG_WRONG, "Quad aspect ratio not yet implemented"); // TODO
    break;
  case 3:
    if (simplex) SETERRQ(comm, PETSC_ERR_ARG_WRONG, "Tet aspect ratio not yet implemented"); // TODO
    else SETERRQ(comm, PETSC_ERR_ARG_WRONG, "Hex aspect ratio not yet implemented"); // TODO
    break;
  default:
    SETERRQ(comm, PETSC_ERR_ARG_WRONG, "Dimension %d is not supported, only 2 or 3", dim);
  }
  PetscFunctionReturn(0);
}
