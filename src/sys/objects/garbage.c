#include <petsc/private/garbagecollector.h>

/* Fetches garbage hashmap from communicator */
static PetscErrorCode GarbageGetHMap_Private(MPI_Comm comm,PetscHMapObj **garbage)
{
  PetscMPIInt    flag;

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

    Petsc objects are given a creation index at initialisation based on
    the communicator it was created on and the order in which it is
    created. When this function is passed a Petsc object a pointer to
    the object is stashed on a garbage dictionary (PetscHMapObj), keyed
    by its creation index.

    Objects stashed on this garbage dictionary can later be destroyed
    with a call to PetscGarbageCleanup().

    This function is intended for use with managed languages such as
    Python or Julia, which may no destroy objects in a deterministic
    order.

    Level: developer

.seealso: PetscGarbageCleanup()
@*/
PetscErrorCode PetscObjectDelayedDestroy(PetscObject *obj)
{
  MPI_Comm       petsc_comm;
  PetscObject    *duplicate;
  PetscHMapObj   *garbage;

  PetscFunctionBegin;
  /* Don't stash NULL pointers */
  if ((*obj != NULL) && (--(*obj)->refct == 0)) {
    (*obj)->refct = 1;
    PetscCall(PetscObjectGetComm(*obj,&petsc_comm));
    PetscCall(GarbageGetHMap_Private(petsc_comm,&garbage));

    /* Duplicate object header so managed language can clean up original */
    PetscCall(PetscNew(&duplicate));
    PetscCall(PetscMemcpy(duplicate,obj,sizeof(PetscObject)));
    PetscCall(PetscHMapObjSet(*garbage,(*duplicate)->cidx,duplicate));
  }
  *obj = NULL;
  PetscFunctionReturn(0);
}

/* Performs the intersection of 2 sorted arrays seta and setb of lengths
   lena and lenb respectively,returning the result in seta and lena
   This is an O(n) operation */
static PetscErrorCode GarbageKeySortedIntersect_Private(PetscInt *seta,PetscInt *lena,PetscInt *setb,PetscInt lenb)
{
  /* The arrays seta and setb MUST be sorted! */
  PetscInt ii,counter = 0;
  PetscInt *endb;

  PetscFunctionBegin;
  endb = setb + lenb;
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
static PetscErrorCode GarbageKeyGatherIntersect_Private(MPI_Comm comm,PetscInt *set,PetscInt *entries)
{
  PetscInt       ii,total;
  PetscInt       *recvset;
  PetscMPIInt    comm_size,comm_rank;
  PetscMPIInt    *set_sizes,*displace;

  PetscFunctionBegin;
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
  PetscCallMPI(MPI_Gatherv(set,*entries,MPI_INT,recvset,set_sizes,displace,MPI_INT,0,comm));
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

/* Calculate block for given communicator and blocksize */
static PetscErrorCode CalculateBlock_Private(MPI_Comm comm,int blocksize)
{
  PetscInt       p,block;
  PetscMPIInt    comm_size;

  PetscFunctionBegin;
  /* Calculate biggest power of blocksize smaller than communicator size */
  PetscCallMPI(MPI_Comm_size(comm,&comm_size));
  p = (PetscInt)PetscFloorReal(PetscLogReal((PetscReal)comm_size)/PetscLogReal((PetscReal)blocksize));
  block = PetscPowInt(blocksize,p);
  if ((block == comm_size) && (comm_size != 1)) {
    block = PetscPowInt(blocksize,(p - 1));
  }
  PetscFunctionReturn(block);
}

/* Setup inter- and intra- communicators for more efficient MPI */
static PetscErrorCode GarbageSetupComms_Private(MPI_Comm comm,int blocksize)
{
  PetscInt       block,blockrank,leader;
  PetscMPIInt    comm_rank,intra_rank;
  MPI_Comm       *intracom,*intercom;

  PetscFunctionBegin;
  /* Calculate biggest power of blocksize smaller than communicator size */
  block = CalculateBlock_Private(comm, blocksize);

  /* Create intra-communicators of size block or smaller */
  PetscCallMPI(MPI_Comm_rank(comm,&comm_rank));
  blockrank = comm_rank/block;
  PetscCall(PetscNew(&intracom));
  PetscCallMPI(MPI_Comm_split(comm,blockrank,comm_rank,intracom));
  PetscCallMPI(MPI_Comm_set_attr(comm,Petsc_Garbage_IntraComm_keyval,(void*)intracom));

  /* Create inter-communicators between rank 0 of all above comms */
  PetscCallMPI(MPI_Comm_rank(*intracom,&intra_rank));
  leader = (intra_rank == 0) ? 0 : MPI_UNDEFINED;
  PetscCall(PetscNew(&intercom));
  PetscCallMPI(MPI_Comm_split(comm,leader,comm_rank,intercom));
  PetscCallMPI(MPI_Comm_set_attr(comm,Petsc_Garbage_InterComm_keyval,(void*)intercom));
  PetscFunctionReturn(0);
}

/*@C
    PetscGarbageCleanup - Destroys objects placed in the garbage by
    PetscObjectDelayedDestroy().

    Collective

    Input Parameters:
+   obj       - communicator over which to perform collective cleanup
-   blocksize - number of MPI ranks to block together for communication

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
PetscErrorCode PetscGarbageCleanup(MPI_Comm comm,PetscInt blocksize)
{
  PetscInt       ii,entries,offset;
  PetscInt       *keys;
  PetscObject    *obj;
  PetscHMapObj   *garbage;
  PetscMPIInt    flag,intra_rank,inter_rank;
  MPI_Comm       *intracom,*intercom;

  PetscFunctionBegin;
  /* Get the garbage hash map */
  PetscCall(PetscCommDuplicate(comm,&comm,NULL));
  PetscCall(GarbageGetHMap_Private(comm,&garbage));

  /* Get the intra- and inter- communicators,if they exist,otherwise set them up */
  intracom = NULL;
  PetscCallMPI(MPI_Comm_get_attr(comm,Petsc_Garbage_IntraComm_keyval,&intracom,&flag));
  if (!flag) {
    PetscCall(GarbageSetupComms_Private(comm,blocksize));
    PetscCallMPI(MPI_Comm_get_attr(comm,Petsc_Garbage_IntraComm_keyval,&intracom,&flag));
  }
  PetscCallMPI(MPI_Comm_get_attr(comm,Petsc_Garbage_InterComm_keyval,&intercom,&flag));
  PetscCallMPI(MPI_Comm_rank(*intracom,&intra_rank));
  if (*intercom != MPI_COMM_NULL) {
    PetscCallMPI(MPI_Comm_rank(*intercom,&inter_rank));
  } else {
    inter_rank = MPI_UNDEFINED;
  }

  /* Get keys from garbage hash map and sort */
  PetscCall(PetscHMapObjGetSize(*garbage,&entries));
  PetscCall(PetscMalloc1(entries,&keys));
  offset = 0;
  PetscCall(PetscHMapObjGetKeys(*garbage,&offset,keys));
  PetscCall(PetscSortInt(entries,keys));

  /* Intracom gather and intersect */
  PetscCall(GarbageKeyGatherIntersect_Private(*intracom,keys,&entries));

  /* Intercom gather and intersect */
  if (*intercom != MPI_COMM_NULL) {
    PetscCall(GarbageKeyGatherIntersect_Private(*intercom,keys,&entries));
    /* Broadcast across intercom */
    PetscCallMPI(MPI_Bcast(&entries,1,MPI_INT,0,*intercom));
    PetscCallMPI(MPI_Bcast(keys,entries,MPI_INT,0,*intercom));
  }

  /* Broadcast across intracom */
  PetscCallMPI(MPI_Bcast(&entries,1,MPI_INT,0,*intracom));
  PetscCallMPI(MPI_Bcast(keys,entries,MPI_INT,0,*intracom));

  /* Collectively destroy objects objects that appear in garbage in
   * creation index order */
  for (ii = 0; ii < entries; ii++) {
    PetscCall(PetscHMapObjGet(*garbage,keys[ii],&obj));
    if (PetscCheckPointer((void*) obj,PETSC_OBJECT) && (obj != NULL)) {
      PetscCall(PetscObjectDestroy(obj));
      PetscCall(PetscFree(obj));
    }
    PetscCall(PetscHMapObjDel(*garbage,keys[ii]));
  }
  PetscCall(PetscFree(keys));
  PetscCall(PetscCommDestroy(&comm));
  PetscFunctionReturn(0);
}

/* Performs a collective intersection of one array per rank _recursively_
   Replaces above GatherIntersect function */
static PetscErrorCode GarbageKeyRecursiveGatherIntersect_Private(MPI_Comm comm,PetscInt *set,PetscInt *entries,PetscInt blocksize)
{
  PetscMPIInt    flag,comm_size,intra_rank,inter_rank;
  MPI_Comm       *intracom,*intercom;

  PetscFunctionBegin;
  PetscCallMPI(MPI_Comm_size(comm,&comm_size));
  if (comm_size > blocksize) {
    /* Get the intra- and inter- communicators,if they exist,otherwise set them up */
    intracom = NULL;
    PetscCallMPI(MPI_Comm_get_attr(comm,Petsc_Garbage_IntraComm_keyval,&intracom,&flag));
    if (!flag) {
      PetscCall(GarbageSetupComms_Private(comm,blocksize));
      PetscCallMPI(MPI_Comm_get_attr(comm,Petsc_Garbage_IntraComm_keyval,&intracom,&flag));
    }
    PetscCallMPI(MPI_Comm_get_attr(comm,Petsc_Garbage_InterComm_keyval,&intercom,&flag));
    PetscCallMPI(MPI_Comm_rank(*intracom,&intra_rank));
    if (*intercom != MPI_COMM_NULL) {
      PetscCallMPI(MPI_Comm_rank(*intercom,&inter_rank));
    } else {
      inter_rank = MPI_UNDEFINED;
    }

    /* Gather and intersect on intracom recursively */
    PetscCall(GarbageKeyRecursiveGatherIntersect_Private(*intracom,set,entries,blocksize));
    /* Gather intersect over intercom */
    if (*intercom != MPI_COMM_NULL) {
      PetscCall(GarbageKeyGatherIntersect_Private(*intercom,set,entries));
    }
  } else {
    PetscCall(GarbageKeyGatherIntersect_Private(comm,set,entries));
  }
  PetscFunctionReturn(0);
}

/* Performs broadcast of resultant intersection _recursively_ */
static PetscErrorCode GarbageKeyRecursiveBcast_Private(MPI_Comm comm,PetscInt *set,PetscInt *entries)
{
  PetscMPIInt flag;
  MPI_Comm    *intracom,*intercom;

  PetscFunctionBegin;
  PetscCallMPI(MPI_Comm_get_attr(comm,Petsc_Garbage_IntraComm_keyval,&intracom,&flag));
  if (flag) {
    PetscCallMPI(MPI_Comm_get_attr(comm,Petsc_Garbage_InterComm_keyval,&intercom,&flag));
    if (*intercom != MPI_COMM_NULL) {
      /* Broadcast across intercom */
      PetscCallMPI(MPI_Bcast(entries,1,MPI_INT,0,*intercom));
      PetscCallMPI(MPI_Bcast(set,*entries,MPI_INT,0,*intercom));
    }
    /* Broadcast across intracom */
    PetscCall(GarbageKeyRecursiveBcast_Private(*intracom,set,entries));
  } else {
    /* If the comm has no intracom,we are at the recursion base case */
    PetscCallMPI(MPI_Bcast(entries,1,MPI_INT,0,comm));
    PetscCallMPI(MPI_Bcast(set,*entries,MPI_INT,0,comm));
  }
  PetscFunctionReturn(0);
}

/*@C
    PetscGarbageRecursiveCleanup - A recursive implementation of
    PetscGarbageCleanup

    Collective

    Input Parameters:
+   obj       - communicator over which to perform collective cleanup
-   blocksize - number of MPI ranks to block together for communication

    Level: developer

.seealso: PetscObjectDelayedDestroy()
@*/
PetscErrorCode PetscGarbageRecursiveCleanup(MPI_Comm comm,PetscInt blocksize)
{
  PetscInt     ii,entries,offset;
  PetscInt     *keys;
  PetscObject  *obj;
  PetscHMapObj *garbage;

  PetscFunctionBegin;
  /* Duplicate comm to prevent it being cleaned up by PetscObjectDestroy() */
  PetscCall(PetscCommDuplicate(comm,&comm,NULL));

  /* Grab garbage from comm and remove it
   this avoids calling PetscCommDestroy() and endlessly recursing */
  PetscCall(GarbageGetHMap_Private(comm,&garbage));
  PetscCallMPI(MPI_Comm_delete_attr(comm,Petsc_Garbage_HMap_keyval));

  /* Get keys from garbage hash map and sort */
  PetscCall(PetscHMapObjGetSize(*garbage,&entries));

  PetscCall(PetscMalloc1(entries,&keys));
  offset = 0;
  PetscCall(PetscHMapObjGetKeys(*garbage,&offset,keys));

  PetscCall(PetscSortInt(entries,keys));

  /* Recursive gather+intersect and broadcast */
  PetscCall(GarbageKeyRecursiveGatherIntersect_Private(comm,keys,&entries,blocksize));
  PetscCall(GarbageKeyRecursiveBcast_Private(comm,keys,&entries));

  for (ii=0; ii<entries; ii++) {
    PetscCall(PetscHMapObjGet(*garbage,keys[ii],&obj));
    if (PetscCheckPointer((void*)obj,PETSC_OBJECT) && (obj != NULL)) {
      PetscCall(PetscObjectDestroy(obj));
      PetscCall(PetscFree(obj));
    }
    PetscCall(PetscHMapObjDel(*garbage,keys[ii]));
  }
  PetscCall(PetscFree(keys));

  /* Put garbage back */
  if (comm != MPI_COMM_NULL) {
    PetscCallMPI(MPI_Comm_set_attr(comm, Petsc_Garbage_HMap_keyval, garbage));
  } else {
    PetscPrintf(comm, "No comm to stash garbage on!!!\n");
  }

  /* Cleanup comm if we made one */
  PetscCall(PetscCommDestroy(&comm));
  PetscFunctionReturn(0);
}

/* Utility function for printing the contents of the garbage on a given comm */
PetscErrorCode PrintGarbage_Private(MPI_Comm comm)
{
  PetscInt     ii,entries,offset;
  PetscInt     *keys;
  PetscObject  *obj;
  PetscHMapObj *garbage;

  PetscFunctionBegin;
  PetscPrintf(comm, "PETSc garbage on ");
  if (comm==PETSC_COMM_WORLD) {
    PetscPrintf(comm, "PETSC_COMM_WORLD, id = %ld\n", comm);
  } else if (comm==PETSC_COMM_SELF) {
    PetscPrintf(comm, "PETSC_COMM_SELF, id= %ld\n", comm);
  } else {
    PetscPrintf(comm, "UNKNOWN_COMM, id = %ld\n", comm);
  }
  PetscCall(PetscCommDuplicate(comm,&comm,NULL));
  PetscCall(GarbageGetHMap_Private(comm,&garbage));

  /* Get keys from garbage hash map and sort */
  PetscCall(PetscHMapObjGetSize(*garbage,&entries));
  PetscPrintf(comm, "Total entries: %i\n", entries);
  PetscCall(PetscMalloc1(entries,&keys));
  offset = 0;
  PetscCall(PetscHMapObjGetKeys(*garbage,&offset,keys));

  /* Pretty print entries in a table */
  if (entries) {
    PetscPrintf(comm, "| Key   | Type             | Name                             | Object ID |\n");
    PetscPrintf(comm, "|-------|------------------|----------------------------------|-----------|\n");
  }
  for (ii=0; ii<entries; ii++) {
    PetscCall(PetscHMapObjGet(*garbage,keys[ii],&obj));
    PetscPrintf(comm, "| %5i | %-16s | %-32s | %5i     |\n", keys[ii], (*obj)->class_name, (*obj)->description, (*obj)->id);
  }

  PetscCall(PetscFree(keys));
  PetscCall(PetscCommDestroy(&comm));
  PetscFunctionReturn(0);
}
