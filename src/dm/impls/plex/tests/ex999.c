static char help[] = "Tests for submesh creation\n\n";

/* TODO
*/

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


testNum 1: overlap = 1 in submesh
=========

overlap 1:
---------

subdim 2: subpoints = {0, 1, 2}


           5--13---6--14--(9)-(18)(10)                  5---13--6---14--9--18--10
           |       |       |       |                    |       |       |       |
rank 0:   15   0   16  1  (19) (2)(20)            -->  15   0   16  1   19  2  20
           |       |       |       |                    |       |       |       |
           3--11---4--12--(7)-(17)(8)                   3---11--4---12--7---17--8


                 (10)-(19)-6--13---7---14--8
                   |       |       |       |
rank 1:          (20) (2)  15  0   16  1   17     -->   subdm is not created
                   |       |       |       |
                  (9)-(18)-3--11---4---12--5


testNum 2: overlap = 0 in submesh (create submesh only with owned cells)
=========

overlap 1:
---------

subdim 2: subpoints = {0, 1, 2}


           5--13---6--14--(9)-(18)(10)                  4--10---5---11--7
           |       |       |       |                    |       |       |
rank 0:   15   0   16  1  (19) (2)(20)            -->  12   0   13  1   14
           |       |       |       |                    |       |       |
           3--11---4--12--(7)-(17)(8)                   2---8---3---9---6


                 (10)-(19)-6--13---7---14--8
                   |       |       |       |
rank 1:          (20) (2)  15  0   16  1   17     -->   subdm is not created
                   |       |       |       |
                  (9)-(18)-3--11---4---12--5

*/

typedef struct {
  PetscInt  testNum;                      /* Test # */
  PetscInt  subdim;                       /* Indicates the mesh to create */
  PetscInt  overlap;                      /* The partition overlap */
} AppCtx;

PetscErrorCode ProcessOptions(MPI_Comm comm, AppCtx *options)
{

  PetscFunctionBegin;
  options->testNum = 0;
  options->subdim  = 2;
  options->overlap = 0;

  PetscOptionsBegin(comm, "", "Meshing Interpolation Test Options", "DMPLEX");
  PetscCall(PetscOptionsBoundedInt("-test_num", "The test #", "ex99.c", options->testNum, &options->testNum, NULL,0));
  PetscCall(PetscOptionsBoundedInt("-subdim", "The mesh to create", "ex99.c", options->subdim, &options->subdim, NULL,0));
  PetscCall(PetscOptionsBoundedInt("-overlap", "The partition overlap", "ex99.c", options->overlap, &options->overlap, NULL,0));
  PetscOptionsEnd();
  PetscFunctionReturn(0);
}

int main(int argc, char **argv)
{
  DM              dm, subdm;
  DMLabel         filter;
  PetscInt        height;
  const PetscInt  filterValue = 1;
  MPI_Comm        comm;
  AppCtx          user;
  PetscMPIInt     size, rank;

  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  comm = PETSC_COMM_WORLD;
  PetscCall(ProcessOptions(comm, &user));
  PetscCallMPI(MPI_Comm_size(comm, &size));
  if (size != 2) {
    PetscCall(PetscPrintf(comm, "This example is specifically designed for size == 2.\n"));
    PetscCall(PetscFinalize());
    return 0;
  }
  PetscCallMPI(MPI_Comm_rank(comm, &rank));
  PetscCall(DMLabelCreate(PETSC_COMM_SELF, "filter", &filter));

  /* Create parallel dm */
  const PetscInt faces[2] = {4,1};
  DM             pdm;
  PetscSF        sf;
  PetscCall(DMPlexCreateBoxMesh(comm, 2, PETSC_FALSE, faces, NULL, NULL, NULL, PETSC_TRUE, &dm));
  PetscCall(DMPlexDistribute(dm, user.overlap, &sf, &pdm));
  if (pdm) {
    PetscCall(DMDestroy(&dm));
    dm = pdm;
  }
  if (sf) {
    PetscCall(PetscSFDestroy(&sf));
  }

  /* Define height */
  height = 2 - user.subdim;

  /* Create filter label */
  switch (user.testNum) {
    case 0:
      switch (user.subdim) {
        case 2:
        {
          switch (user.overlap) {
            case 0:
              if (rank==0) {
                DMLabelSetValue(filter, 0, filterValue);
                DMLabelSetValue(filter, 1, filterValue);
              }
              break;
            case 1:
              if (rank==0) {
                DMLabelSetValue(filter, 0, filterValue);
                DMLabelSetValue(filter, 1, filterValue);
              } else if (rank==1) {
                DMLabelSetValue(filter, 2, filterValue);
                DMLabelSetValue(filter, 1, filterValue);
              }
              break;
          }
          break;
        }
        case 1:
        {
          switch (user.overlap) {
            case 0:
              if (rank==0) {
                DMLabelSetValue(filter, 10, filterValue);
                DMLabelSetValue(filter, 11, filterValue);
              }
              break;
            case 1:
              if (rank==0) {
                DMLabelSetValue(filter, 13, filterValue);
                DMLabelSetValue(filter, 14, filterValue);
              } else if (rank==1) {
                DMLabelSetValue(filter, 19, filterValue);
                DMLabelSetValue(filter, 14, filterValue);
              }
              break;
          }
          break;
        }
        case 0:
        {
          switch (user.overlap) {
            case 0:
              if (rank==0) {
                DMLabelSetValue(filter, 5, filterValue);
                DMLabelSetValue(filter, 6, filterValue);
                DMLabelSetValue(filter, 7, filterValue);
              } else if (rank==1) {
                DMLabelSetValue(filter, 5, filterValue);
              }
              break;
            case 1:
              if (rank==0) {
                DMLabelSetValue(filter, 5, filterValue);
                DMLabelSetValue(filter, 6, filterValue);
                DMLabelSetValue(filter, 9, filterValue);
              } else if (rank==1) {
                DMLabelSetValue(filter, 6, filterValue);
                DMLabelSetValue(filter, 8, filterValue);
              }
              break;
          }
          break;
        }
      }
      break;
    case 1:
    case 2:
      if (rank==0) {
        DMLabelSetValue(filter, 0, filterValue);
        DMLabelSetValue(filter, 1, filterValue);
        DMLabelSetValue(filter, 2, filterValue);
      }
      break;
  }
  PetscCall(PetscObjectSetName((PetscObject) dm, "Example_DM"));
  PetscCall(DMViewFromOptions(dm, NULL, "-dm_view"));
  switch (user.testNum) {
    case 0:
      PetscCall(DMPlexCreateSubmeshGeneric(dm, DMPLEX_SUBMESH_CLOSURE, filter, filterValue, height, PETSC_FALSE, PETSC_FALSE, PETSC_FALSE, NULL, NULL, PETSC_FALSE, &subdm));
      PetscCall(PetscObjectSetName((PetscObject) subdm, "Example_SubDM"));
      PetscCall(DMViewFromOptions(subdm, NULL, "-dm_view"));
      PetscCall(DMDestroy(&subdm));
      break;
    case 1:
    /* Submesh only on rank == 0 */
      if (rank == 0) {
        PetscCall(DMPlexCreateSubmeshGeneric(dm, DMPLEX_SUBMESH_CLOSURE, filter, filterValue, height, PETSC_FALSE, PETSC_FALSE, PETSC_FALSE, NULL, NULL, PETSC_TRUE, &subdm));
        PetscCall(PetscObjectSetName((PetscObject) subdm, "Example_SubDM_Local"));
        PetscCall(DMViewFromOptions(subdm, NULL, "-dm_view"));
        PetscCall(DMDestroy(&subdm));
      }
      break;
    case 2:
    /* Submesh only on rank == 0. Use different adjacency overlap */
    {
      /* Create new parent dm with overlap = 0 */
/*      DM dmCopy;
      PetscCall(DMClone(dm, &dmCopy));
      PetscCall(DMPlexSetOverlap_Plex(dmCopy, NULL, 0));
      if (rank == 0) {
        PetscCall(DMPlexCreateSubmeshGeneric(dmCopy, DMPLEX_SUBMESH_CLOSURE, filter, filterValue, height, PETSC_FALSE, PETSC_FALSE, PETSC_FALSE, NULL, NULL, PETSC_TRUE, &subdm));
        PetscCall(PetscObjectSetName((PetscObject) subdm, "Example_SubDM_Local"));
        PetscCall(DMViewFromOptions(subdm, NULL, "-dm_view"));
        PetscCall(DMDestroy(&subdm));
      }
      PetscCall(DMDestroy(&dmCopy));
      break;*/
    }
  }
  PetscCall(DMLabelDestroy(&filter));
  PetscCall(DMDestroy(&dm));

  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

  # Four cell tests
  test:
    suffix: 0
    nsize: 2
    args: -test_num 0 -subdim 2 -overlap 0 -dm_view ascii::ascii_info_detail
  test:
    suffix: 1
    nsize: 2
    args: -test_num 0 -subdim 1 -overlap 0 -dm_view ascii::ascii_info_detail
  test:
    suffix: 2
    nsize: 2
    args: -test_num 0 -subdim 0 -overlap 0 -dm_view ascii::ascii_info_detail
  test:
    suffix: 3
    nsize: 2
    args: -test_num 0 -subdim 2 -overlap 1 -dm_view ascii::ascii_info_detail
  test:
    suffix: 4
    nsize: 2
    args: -test_num 0 -subdim 1 -overlap 1 -dm_view ascii::ascii_info_detail
  test:
    suffix: 5
    nsize: 2
    args: -test_num 0 -subdim 0 -overlap 1 -dm_view ascii::ascii_info_detail
  test:
    suffix: 6
    nsize: 2
    args: -test_num 1 -subdim 2 -overlap 1 -dm_view ascii::ascii_info_detail
#  test:
#    suffix: 7
#    nsize: 2
#    args: -test_num 2 -subdim 2 -overlap 1 -dm_view ascii::ascii_info_detail

TEST*/
