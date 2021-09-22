static char help[] = "Tests mesh quality functions.\n";

#include <petsc/private/dmpleximpl.h>

int main(int argc, char **argv) {
  DM              dm, dmDist;
  MPI_Comm        comm;
  PetscBool       simplex = PETSC_FALSE;
  PetscErrorCode  ierr;
  PetscInt       *faces, dim = 2, numEdges = 4, d;
  PetscReal       err, tol = 1.0e-4;
  Vec             aspectRatio, expected;

  /* Set up */
  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  comm = PETSC_COMM_WORLD;
  ierr = PetscOptionsBegin(comm, "", "Mesh adaptation options", "DMPLEX");PetscCall(ierr);
  PetscCall(PetscOptionsRangeInt("-dim", "The topological mesh dimension", "ex43.c", dim, &dim, NULL, 2, 3));
  PetscCall(PetscOptionsInt("-num_edges", "Number of edges on each boundary of the initial mesh", "ex43.c", numEdges, &numEdges, NULL));
  PetscCall(PetscOptionsBool("-simplex", "Simplex elements?", "ex43.c", simplex, &simplex, NULL));
  ierr = PetscOptionsEnd();

  /* Create box mesh */
  PetscCall(PetscMalloc1(dim, &faces));
  for (d = 0; d < dim; ++d) faces[d] = numEdges;
  PetscCall(DMPlexCreateBoxMesh(comm, dim, simplex, faces, NULL, NULL, NULL, PETSC_TRUE, &dm));
  PetscCall(PetscFree(faces));

  /* Distribute mesh over processes */
  PetscCall(DMPlexDistribute(dm, 0, NULL, &dmDist));
  if (dmDist) {
    PetscCall(DMDestroy(&dm));
    dm = dmDist;
  }
  PetscCall(DMViewFromOptions(dm, NULL, "-dm_view"));

  /* Compute aspect ratio */
  PetscCall(DMPlexGetAspectRatio(dm, &aspectRatio));
  PetscCall(VecViewFromOptions(aspectRatio, NULL, "-aspect_ratio_view"));
  PetscCall(VecDuplicate(aspectRatio, &expected));
  PetscCall(VecSet(expected, 1.207107));  // TODO: switch dim and simplex cases
  PetscCall(VecAXPY(expected, -1, aspectRatio));
  PetscCall(VecNorm(expected, NORM_2, &err));
  PetscCheck(err <= tol, comm, PETSC_ERR_ARG_OUTOFRANGE, "Aspect ratio does not match expected value (L2 error %f)", err);

  // TODO: other quality measures

  /* Clean up */
  PetscCall(VecDestroy(&expected));
  PetscCall(VecDestroy(&aspectRatio));
  PetscCall(DMDestroy(&dm));
  PetscFinalize();
  return 0;
}
