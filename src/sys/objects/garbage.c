#include <petsc/private/garbagecollector.h>

/* Fetches garbage hashmap from communicator */
static PetscErrorCode GarbageGetHMap_Private(MPI_Comm comm,PetscHMapObj **garbage)
{
  PetscMPIInt flag;

  PetscFunctionBegin;
  PetscCallMPI(MPI_Comm_get_attr(comm,Petsc_Garbage_HMap_keyval,garbage,&flag));
  if (!flag) {
    /* No garbage,create one */
    PetscCall(PetscNew(garbage));
    PetscCall(PetscHMapObjCreate(*garbage));
    PetscCallMPI(MPI_Comm_set_attr(comm,Petsc_Garbage_HMap_keyval,*garbage));
  }
  PetscFunctionReturn(0);
}

/*@C
    PetscObjectDelayedDestroy - Adds an object to a data structure for
    later destruction.

    Not Collective

    Input Parameters:
.   obj - object to be destroyed

    Notes: Analogue to PetscObjectDestroy() for use in managed languages.

    A Petsc object is given a creation index at initialisation based on
    the communicator it was created on and the order in which it is
    created. When this function is passed a Petsc object, a pointer to
    the object is stashed on a garbage dictionary (PetscHMapObj) which is
    keyed by its creation index.

    Objects stashed on this garbage dictionary can later be destroyed
    with a call to PetscGarbageCleanup().

    This function is intended for use with managed languages such as
    Python or Julia, which may not destroy objects in a deterministic
    order.

    Level: developer

.seealso: PetscGarbageCleanup()
@*/
PetscErrorCode PetscObjectDelayedDestroy(PetscObject *obj)
{
  MPI_Comm     petsc_comm;
  PetscHMapObj *garbage;

  PetscFunctionBegin;
  /* Don't stash NULL pointers */
  if ((*obj != NULL) && (--(*obj)->refct == 0)) {
    (*obj)->refct = 1;
    PetscCall(PetscObjectGetComm(*obj,&petsc_comm));
    PetscCall(GarbageGetHMap_Private(petsc_comm,&garbage));
    PetscCall(PetscHMapObjSet(*garbage,(*obj)->cidx,*obj));
  }
  *obj = NULL;
  PetscFunctionReturn(0);
}

/* Performs the intersection of 2 sorted arrays seta and setb of lengths
   lena and lenb respectively,returning the result in seta and lena
   This is an O(n) operation */
static PetscErrorCode GarbageKeySortedIntersect_Private(PetscCount seta[],PetscInt *lena,PetscCount setb[],PetscInt lenb)
{
  /* The arrays seta and setb MUST be sorted! */
  PetscInt   ii,counter = 0;
  PetscCount *endb;

  PetscFunctionBegin;
  endb = setb + (PetscCount)lenb;
  for (ii=0; ii<*lena; ii++) {
    while ((seta[ii] > *setb) && (setb < endb)) {
      setb++;
    }
    if (seta[ii] == *setb) {
      seta[counter] = seta[ii];
      counter++;
    }
  }
  *lena = counter;
  PetscFunctionReturn(0);
}

/* Performs a collective intersection of one array per rank */
static PetscErrorCode GarbageKeyGatherIntersect_Private(MPI_Comm comm,PetscCount *set,PetscInt *entries)
{
  PetscInt    ii,total;
  PetscCount  *recvset;
  PetscMPIInt comm_size,comm_rank;
  PetscMPIInt *set_sizes,*displace;

  PetscFunctionBegin;
  /* Sort keys first for use with `GarbageKeySortedIntersect_Private()`*/
  PetscCall(PetscSortCount(*entries,set));

  /* Gather and intersect on comm */
  PetscCallMPI(MPI_Comm_size(comm,&comm_size));
  PetscCallMPI(MPI_Comm_rank(comm,&comm_rank));
  if (comm_rank == 0) {
    PetscCall(PetscMalloc1(comm_size,&set_sizes));
  } else {
    set_sizes = NULL;
  }

  /* Gather number of keys from each rank */
  PetscCallMPI(MPI_Gather(entries,1,MPI_INT,set_sizes,1,MPI_INT,0,comm));
  if (comm_rank == 0) {
    PetscCall(PetscMalloc1(comm_size,&displace));
    displace[0] = 0;
    total = 0;
    for (ii=1; ii<comm_size; ii++) {
      total += set_sizes[ii - 1];
      displace[ii] = total;
    }
    total += set_sizes[comm_size - 1];
    PetscCall(PetscMalloc1(total,&recvset));
  } else {
    displace = NULL;
    recvset = NULL;
  }

  /* Gatherv keys from all ranks and intersect */
  PetscCallMPI(MPI_Gatherv(set,*entries,MPI_AINT,recvset,set_sizes,displace,MPI_AINT,0,comm));
  if (comm_rank == 0) {
    for (ii=1; ii<comm_size; ii++) {
      PetscCall(GarbageKeySortedIntersect_Private(set,entries,&recvset[displace[ii]],set_sizes[ii]));
    }
  }

  PetscCall(PetscFree(set_sizes));
  PetscCall(PetscFree(displace));
  PetscCall(PetscFree(recvset));
  PetscFunctionReturn(0);
}

/* Setup inter- and intra- communicators for more efficient MPI */
static PetscErrorCode GarbageSetupComms_Private(MPI_Comm comm)
{
  PetscInt    leader;
  PetscMPIInt comm_rank,intra_rank;
  MPI_Comm    *intracomm,*intercomm;

  PetscFunctionBegin;
  /* Create shared memory intra-communicators */
  PetscCallMPI(MPI_Comm_rank(comm,&comm_rank));
  PetscCall(PetscNew(&intracomm));
  PetscCallMPI(MPI_Comm_split(comm,MPI_COMM_TYPE_SHARED,comm_rank,intracomm));
  PetscCallMPI(MPI_Comm_set_attr(comm,Petsc_Garbage_IntraComm_keyval,(void*)intracomm));

  /* Create inter-communicators between rank 0 of all above comms */
  PetscCallMPI(MPI_Comm_rank(*intracomm,&intra_rank));
  leader = (intra_rank == 0) ? 0 : MPI_UNDEFINED;
  PetscCall(PetscNew(&intercomm));
  PetscCallMPI(MPI_Comm_split(comm,leader,comm_rank,intercomm));
  PetscCallMPI(MPI_Comm_set_attr(comm,Petsc_Garbage_InterComm_keyval,(void*)intercomm));
  PetscFunctionReturn(0);
}

/*@C
    PetscGarbageCleanup - Destroys objects placed in the garbage by
    PetscObjectDelayedDestroy().

    Collective

    Input Parameters:
.   comm      - communicator over which to perform collective cleanup

    Notes: Implements a collective garbage collection.
    A per- MPI communicator garbage dictionary is created to store
    references to objects destroyed using PetscObjectDelayedDestroy().
    Objects that appear in this dictionary on all ranks can be destroyed
    by calling PetscGarbageCleanup().

    This is done as follows:
    1.  Keys of the garbage dictionary, which correspond to the creation
        indices of the objects stashed, are sorted.
    2.  A collective intersection of dictionary keys is performed by all
        ranks in the communicator.
    3.  The intersection is broadcast back to all ranks in the
        communicator.
    4.  The objects on the dictionary are collectively destroyed in
        creation index order using a call to PetscObjectDestroy().

    This function is intended for use with managed languages such as
    Python or Julia, which may no destroy objects in a deterministic
    order.

    Level: developer

.seealso: PetscObjectDelayedDestroy()
@*/
PetscErrorCode PetscGarbageCleanup(MPI_Comm comm)
{
  PetscInt     ii,entries,offset;
  PetscCount   *keys;
  PetscObject  obj;
  PetscHMapObj *garbage;
  PetscMPIInt  flag,intra_rank,inter_rank;
  MPI_Comm     *intracomm,*intercomm;

  PetscFunctionBegin;
  /* Duplicate comm to prevent it being cleaned up by PetscObjectDestroy() */
  PetscCall(PetscCommDuplicate(comm,&comm,NULL));

  /* Grab garbage from comm and remove it
   this avoids calling PetscCommDestroy() and endlessly recursing */
  PetscCall(GarbageGetHMap_Private(comm,&garbage));
  PetscCallMPI(MPI_Comm_delete_attr(comm,Petsc_Garbage_HMap_keyval));

  /* Get the intra- and inter- communicators,if they exist,otherwise set them up */
  intracomm = NULL;
  PetscCallMPI(MPI_Comm_get_attr(comm,Petsc_Garbage_IntraComm_keyval,&intracomm,&flag));
  if (!flag) {
    PetscCall(GarbageSetupComms_Private(comm));
    PetscCallMPI(MPI_Comm_get_attr(comm,Petsc_Garbage_IntraComm_keyval,&intracomm,&flag));
  }
  PetscCallMPI(MPI_Comm_get_attr(comm,Petsc_Garbage_InterComm_keyval,&intercomm,&flag));
  PetscCallMPI(MPI_Comm_rank(*intracomm,&intra_rank));
  if (*intercomm != MPI_COMM_NULL) {
    PetscCallMPI(MPI_Comm_rank(*intercomm,&inter_rank));
  } else {
    inter_rank = MPI_UNDEFINED;
  }

  /* Get keys from garbage hash map */
  PetscCall(PetscHMapObjGetSize(*garbage,&entries));
  PetscCall(PetscMalloc1(entries,&keys));
  offset = 0;
  PetscCall(PetscHMapObjGetKeys(*garbage,&offset,keys));

  /* intracomm gather and intersect */
  PetscCall(GarbageKeyGatherIntersect_Private(*intracomm,keys,&entries));

  /* intercomm gather and intersect */
  if (*intercomm != MPI_COMM_NULL) {
    PetscCall(GarbageKeyGatherIntersect_Private(*intercomm,keys,&entries));
    /* Broadcast across intercomm */
    PetscCallMPI(MPI_Bcast(&entries,1,MPIU_INT,0,*intercomm));
    PetscCallMPI(MPI_Bcast(keys,entries,MPIU_INT,0,*intercomm));
  }

  /* Broadcast across intracomm */
  PetscCallMPI(MPI_Bcast(&entries,1,MPIU_INT,0,*intracomm));
  PetscCallMPI(MPI_Bcast(keys,entries,MPIU_INT,0,*intracomm));

  /* Collectively destroy objects objects that appear in garbage in
     creation index order */
  for (ii = 0; ii < entries; ii++) {
    PetscCall(PetscHMapObjGet(*garbage,keys[ii],&obj));
    PetscCall(PetscObjectDestroy(&obj));
    PetscCall(PetscFree(obj));
    PetscCall(PetscHMapObjDel(*garbage,keys[ii]));
  }
  PetscCall(PetscFree(keys));

  /* Put garbage back */
  PetscCallMPI(MPI_Comm_set_attr(comm,Petsc_Garbage_HMap_keyval,garbage));
  PetscCall(PetscCommDestroy(&comm));
  PetscFunctionReturn(0);
}

/* Utility function for printing the contents of the garbage on a given comm */
PetscErrorCode PrintGarbage_Private(MPI_Comm comm)
{
  char         text[64];
  PetscInt     ii,entries,offset;
  PetscCount   *keys;
  PetscObject  obj;
  PetscHMapObj *garbage;
  PetscMPIInt  rank;

  PetscFunctionBegin;
  PetscPrintf(comm,"PETSc garbage on ");
  if (comm == PETSC_COMM_WORLD) {
    PetscCall(PetscPrintf(comm,"PETSC_COMM_WORLD\n"));
  } else if (comm == PETSC_COMM_SELF) {
    PetscCall(PetscPrintf(comm,"PETSC_COMM_SELF\n"));
  } else {
    PetscCall(PetscPrintf(comm,"UNKNOWN_COMM\n"));
  }
  PetscCall(PetscCommDuplicate(comm,&comm,NULL));
  PetscCall(GarbageGetHMap_Private(comm,&garbage));

  /* Get keys from garbage hash map and sort */
  PetscCall(PetscHMapObjGetSize(*garbage,&entries));
  PetscCall(PetscMalloc1(entries,&keys));
  offset = 0;
  PetscCall(PetscHMapObjGetKeys(*garbage,&offset,keys));

  /* Pretty print entries in a table */
  PetscCallMPI(MPI_Comm_rank(comm,&rank));
  PetscCall(PetscSynchronizedPrintf(comm,"Rank %i:: ", rank));
  PetscCall(PetscFormatConvert("Total entries: %D\n",text));
  PetscCall(PetscSynchronizedPrintf(comm,text,entries));
  if (entries) {
    PetscCall(PetscSynchronizedPrintf(comm,"| Key   | Type             | Name                             | Object ID |\n"));
    PetscCall(PetscSynchronizedPrintf(comm,"|-------|------------------|----------------------------------|-----------|\n"));
  }
  for (ii=0; ii<entries; ii++) {
    PetscCall(PetscHMapObjGet(*garbage,keys[ii],&obj));
    PetscCall(PetscFormatConvert("| %5D | %-16s | %-32s | %5D     |\n",text));
    PetscCall(PetscSynchronizedPrintf(comm,text,keys[ii],obj->class_name,obj->description,obj->id));
  }
  PetscCall(PetscSynchronizedFlush(comm,PETSC_STDOUT));

  PetscCall(PetscFree(keys));
  PetscCall(PetscCommDestroy(&comm));
  PetscFunctionReturn(0);
}
