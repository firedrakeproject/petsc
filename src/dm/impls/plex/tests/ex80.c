static char help[] = "Tests for submesh creation\n\n";

#include <petscdmplex.h>
#include <petscsf.h>

/* Submesh of a four-element mesh

Check if it does:

* transfer ownership when necessary
* remove redundant points that are in the halo in the original mesh, but not in the submesh
* treat disconnected submesh

Global numbering:

     (6)(16)-(7)(17)-(8)       5--14---6--15---7      (10)(20)(11)(21)(12)
      |       |       |        |       |       |        |       |       |
    (18) (1)(19) (2)(20)      16   0  17   1  18      (22) (2)(23) (3)(24)
      |       |       |        |       |       |        |       |       |
     (5)(15)(11)(22)(12)       4--13-(11)(22)(12)      (9)(19)--6--14---7
      |       |       |        |       |       |        |       |       |
     14   0 (23) (3)(24)     (20) (2)(23) (3)(24)     (18) (1) 15   0  16
      |       |       |        |       |       |        |       |       |
      4--13--(9)(21)(10)      (8)(19)-(9)(21)(10)      (8)(17)--4--13---5


     (4)(11)-(5)               3---9---4
      |       |                |       |
    (12) (1)(13)              10   0  11
      |       |                |       |
     (3)(10) (7)               2---8---7
      |       |                |       |
      9   0 (14)             (13) (1) 14
      |       |                |       |
      2---8---6               (5)(12)--6


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

*/

typedef struct {
  PetscBool ignoreLabelHalo; /* Ignore filter values in the halo. */
} AppCtx;

PetscErrorCode ProcessOptions(AppCtx *options)
{
  PetscFunctionBegin;
  options->ignoreLabelHalo = PETSC_FALSE;

  PetscOptionsBegin(PETSC_COMM_SELF, "", "Filtering Problem Options", "DMPLEX");
  PetscCall(PetscOptionsBool("-ignore_label_halo", "Ignore filter values in the halo", "ex80.c", options->ignoreLabelHalo, &options->ignoreLabelHalo, NULL));
  PetscOptionsEnd();
  PetscFunctionReturn(PETSC_SUCCESS);
}
int main(int argc, char **argv)
{
  DM             dm, subdm;
  PetscSF        ownershipTransferSF;
  DMLabel        filter;
  const PetscInt filterValue = 1;
  MPI_Comm       comm;
  PetscMPIInt    size, rank;
  AppCtx         user;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  PetscCall(ProcessOptions(&user));
  comm = PETSC_COMM_WORLD;
  PetscCallMPI(MPI_Comm_size(comm, &size));
  if (size != 3) {
    PetscCall(PetscPrintf(comm, "This example is specifically designed for comm size == 3.\n"));
    PetscCall(PetscFinalize());
    return 0;
  }
  PetscCallMPI(MPI_Comm_rank(comm, &rank));
  {
    DM               pdm;
    const PetscInt   faces[2] = {2, 2};
    PetscPartitioner part;
    PetscInt         overlap = 1;

    PetscCall(DMPlexCreateBoxMesh(comm, 2, PETSC_FALSE, faces, NULL, NULL, NULL, PETSC_TRUE, &dm));
    PetscCall(DMPlexGetPartitioner(dm, &part));
    PetscCall(PetscPartitionerSetFromOptions(part));
    PetscCall(DMSetAdjacency(dm, -1, PETSC_FALSE, PETSC_TRUE));
    PetscCall(DMPlexDistribute(dm, overlap, NULL, &pdm));
    if (pdm) {
      PetscCall(DMDestroy(&dm));
      dm = pdm;
    }
  }
  PetscCall(DMLabelCreate(PETSC_COMM_SELF, "filter", &filter));
  switch (rank) {
  case 0:
    DMLabelSetValue(filter, 0, filterValue);
    DMLabelSetValue(filter, 1, filterValue);
    break;
  case 1:
    DMLabelSetValue(filter, 0, filterValue);
    DMLabelSetValue(filter, 2, filterValue);
    break;
  case 2:
    break;
  }
  PetscCall(PetscObjectSetName((PetscObject)dm, "Example_DM"));
  PetscCall(DMViewFromOptions(dm, NULL, "-dm_view"));
  PetscCall(DMPlexFilter(dm, filter, filterValue, &ownershipTransferSF, &subdm));
  PetscCall(PetscObjectSetName((PetscObject)ownershipTransferSF, "Ownership Transfer SF"));
PetscSFView(ownershipTransferSF, NULL);
  PetscCall(PetscObjectSetName((PetscObject)subdm, "Example_SubDM"));
  PetscCall(DMViewFromOptions(subdm, NULL, "-dm_view"));
  PetscCall(DMDestroy(&subdm));
  PetscCall(PetscSFDestroy(&ownershipTransferSF));
  PetscCall(DMLabelDestroy(&filter));
  PetscCall(DMDestroy(&dm));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

    test:
      suffix: 0
      nsize: 3
      requires: chaco
      args: -petscpartitioner_type chaco -ignore_label_halo -dm_view ascii::ascii_info_detail

TEST*/
