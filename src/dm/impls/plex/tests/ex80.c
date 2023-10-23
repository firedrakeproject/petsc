static char help[] = "Tests for submesh creation\n\n";

#include <petscdmplex.h>
#include <petscsf.h>

/* Submesh of a four-element mesh

Check if it does:

* transfer ownership when necessary
* remove redundant points that are in the halo in the original mesh, but not in the submesh
* treat disconnected submesh

Global numbering:

           6--16---7---17--11--22--12--23--13
           |       |       |       |       |
          18   0   19   1  24  2   25   3  26
           |       |       |       |       |
           4--14---5---15--8---20--9---21--10

* all subpoints = {...} defined below are given in the global numbering

Use two processes throughout

testNum 0: Various submeshes of different dimensions on PETSC_COMM_WORLD
testNum 1: Cell submesh on rank 0
testNum 2: Cell submesh on rank 0 with partition overlap=0

                            dm                                        subdm
                         -------                                    ---------
testNum 0:
=========

overlap 0:
---------

subdim 2: subpoints = {0, 1}

           5--10---6---11-(7)                           5--10---6---11--7
           |       |       |                            |       |       |
rank 0:   12   0   13  1  (14)                    -->  12   0   13  1   14
           |       |       |                            |       |       |
           2---8---3---9--(4)                           2---8---3---9---4

                           5--10---6---11---7
                           |       |        |
rank 1:                   12   0   13   1   14    -->   None
                           |       |        |
                           2---8---3----9---4

subdim 1: subpoints = {16, 17}

           5--10---6---11-(7)                           2---0---3---1---4
           |       |       |
rank 0:   12   0   13  1  (14)                    -->
           |       |       |
           2---8---3---9--(4)

                           5--10---6---11---7
                           |       |        |
rank 1:                   12   0   13   1   14    -->   None
                           |       |        |
                           2---8---3----9---4

subdim 0: subpoints = {6, 7, 11}

           5--10---6---11-(7)                           0       1
           |       |       |
rank 0:   12   0   13  1  (14)                    -->
           |       |       |
           2---8---3---9--(4)

                           5--10---6---11---7                           0
                           |       |        |
rank 1:                   12   0   13   1   14    -->
                           |       |        |
                           2---8---3----9---4

overlap 1:
---------

subdim 2: subpoints = {0, 1, 3}

           5--13---6--14--(9)-(18)(10)                  4--10---5---11--7
           |       |       |       |                    |       |       |
rank 0:   15   0   16  1  (19) (2)(20)            -->  12   0   13  1   14
           |       |       |       |                    |       |       |
           3--11---4--12--(7)-(17)(8)                   2---8---3---9---6

                 (10)-(19)-6--13---7---14--8                                      3---6---4
                   |       |       |       |                                      |       |
rank 1:          (20) (2)  15  0   16  1   17     -->                             7   0   8
                   |       |       |       |                                      |       |
                  (9)-(18)-3--11---4---12--5                                      1 --5---2

subdim 1: subpoints = {16, 17, 23}

           5--13---6--14--(9)-(18)(10)                  2---0---3---1---4
           |       |       |       |
rank 0:   15   0   16  1  (19) (2)(20)            -->
           |       |       |       |
           3--11---4--12--(7)-(17)(8)

                 (10)-(19)-6--13---7---14--8                                      1---0---2
                   |       |       |       |
rank 1:          (20) (2)  15  0   16  1   17     -->
                   |       |       |       |
                  (9)-(18)-3--11---4---12--5

suddim 0: subpoints = {6, 7, 11, 13}

           5--13---6--14--(9)-(18)(10)                  0       1
           |       |       |       |
rank 0:   15   0   16  1  (19) (2)(20)            -->
           |       |       |       |
           3--11---4--12--(7)-(17)(8)

                 (10)-(19)-6--13---7---14--8                            0                 1
                   |       |       |       |
rank 1:          (20) (2)  15  0   16  1   17     -->
                   |       |       |       |
                  (9)-(18)-3--11---4---12--5
*/

typedef struct {
  PetscInt testNum; /* Test # */
} AppCtx;

PetscErrorCode ProcessOptions(MPI_Comm comm, AppCtx *options)
{
  PetscFunctionBegin;
  options->testNum = 0;

  PetscOptionsBegin(comm, "", "Submesh Test Options", "DMPLEX");
  PetscCall(PetscOptionsBoundedInt("-test_num", "The test #", "ex80.c", options->testNum, &options->testNum, NULL, 0));
  PetscOptionsEnd();
  PetscFunctionReturn(0);
}

int main(int argc, char **argv)
{
  DM             dm, subdm;
  DMLabel        filter;
  const PetscInt filterValue = 1;
  MPI_Comm       comm;
  PetscMPIInt    size, rank;
  AppCtx         user;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  comm = PETSC_COMM_WORLD;
  PetscCall(ProcessOptions(comm, &user));
  PetscCallMPI(MPI_Comm_size(comm, &size));
  if (size != 3) {
    PetscCall(PetscPrintf(comm, "This example is specifically designed for size == 3.\n"));
    PetscCall(PetscFinalize());
    return 0;
  }
  PetscCallMPI(MPI_Comm_rank(comm, &rank));
  PetscCall(DMLabelCreate(PETSC_COMM_SELF, "filter", &filter));
  /* Create parallel dm */
  {
    DM             pdm;
    PetscSF        sf;
    const PetscInt faces[2] = {3, 2};
    PetscInt       overlap = 1;

    PetscCall(DMPlexCreateBoxMesh(comm, 2, PETSC_FALSE, faces, NULL, NULL, NULL, PETSC_TRUE, &dm));
    {
      PetscPartitioner part;

      PetscCall(DMPlexGetPartitioner(dm, &part));
      PetscCall(PetscPartitionerSetType(part, PETSCPARTITIONERPARMETIS));
    }
    PetscCall(DMSetAdjacency(dm, -1, PETSC_FALSE, PETSC_TRUE));
    PetscCall(DMPlexDistribute(dm, overlap, &sf, &pdm));
    if (pdm) {
      PetscCall(DMDestroy(&dm));
      dm = pdm;
    }
    if (sf) { PetscCall(PetscSFDestroy(&sf)); }
  }
  /* Create filter label */
/*  switch (user.testNum) {
  case 0:
    switch (user.subdim) {
    case 2: {
      switch (user.overlap) {
      case 0:
        if (rank == 0) {
          DMLabelSetValue(filter, 0, filterValue);
          DMLabelSetValue(filter, 1, filterValue);
        }
        break;
      case 1:
        if (rank == 0) {
          DMLabelSetValue(filter, 0, filterValue);
          DMLabelSetValue(filter, 1, filterValue);
        } else if (rank == 1) {
          DMLabelSetValue(filter, 2, filterValue);
          DMLabelSetValue(filter, 1, filterValue);
        }
        break;
      }
      break;
    }
    }
    break;
  }
*/
  PetscCall(PetscObjectSetName((PetscObject)dm, "Example_DM"));
  PetscCall(DMViewFromOptions(dm, NULL, "-dm_view"));
/*  switch (user.testNum) {
  case 0:
    PetscCall(DMPlexFilter(dm, filter, filterValue, PETSC_TRUE, PETSC_TRUE, PETSC_FALSE, PETSC_TRUE, NULL, NULL, &subdm));
    PetscCall(PetscObjectSetName((PetscObject)subdm, "Example_SubDM"));
    PetscCall(DMViewFromOptions(subdm, NULL, "-dm_view"));
    PetscCall(DMDestroy(&subdm));
    break;
  }
*/
  PetscCall(DMLabelDestroy(&filter));
  PetscCall(DMDestroy(&dm));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

  # Four cell tests
  testset:
    nsize: 3
    requires: parmetis
    args: -dm_view ascii::ascii_info_detail

    test:
      suffix: 0
      args: -test_num 0

TEST*/
