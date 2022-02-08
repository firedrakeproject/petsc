#include "petsc/private/garbagecollector.h"

/* Fetches garbage hashmap from communicator */
static PetscErrorCode GarbageGetHMap_Private(MPI_Comm comm,PetscHMapObj **garbage)
{
  PetscInt       flag;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MPI_Comm_get_attr(comm,Petsc_Garbage_HMap_keyval,garbage,&flag);CHKERRMPI(ierr);
  if (!flag) {
    /* No garbage,create one */
    ierr = PetscNew(garbage);CHKERRQ(ierr);
    ierr = PetscHMapObjCreate(*garbage);CHKERRQ(ierr);
    ierr = MPI_Comm_set_attr(comm,Petsc_Garbage_HMap_keyval,*garbage);CHKERRMPI(ierr);
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
  PetscErrorCode ierr;
  PetscObject    *duplicate;
  PetscHMapObj   *garbage;

  PetscFunctionBegin;
  /* Don't stash NULL pointers */
  if (*obj != NULL) {
    ierr = PetscObjectGetComm(*obj,&petsc_comm);CHKERRQ(ierr);
    ierr = GarbageGetHMap_Private(petsc_comm,&garbage);CHKERRQ(ierr);

    /* Duplicate object header so managed language can clean up original */
    ierr = PetscNew(&duplicate);CHKERRQ(ierr);
    ierr = PetscMemcpy(duplicate,obj,sizeof(PetscObject));CHKERRQ(ierr);
    ierr = PetscHMapObjSet(*garbage,(*duplicate)->cidx,duplicate);CHKERRQ(ierr);
    *obj = NULL;
  }
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
  PetscErrorCode ierr;
  PetscInt       ii,comm_size,comm_rank,total;
  PetscInt       *set_sizes,*displace,*recvset;

  PetscFunctionBegin;
  /* Gather and intersect on comm */
  ierr = MPI_Comm_size(comm,&comm_size);CHKERRMPI(ierr);
  ierr = MPI_Comm_rank(comm,&comm_rank);CHKERRMPI(ierr);
  if (comm_rank == 0) {
    ierr = PetscMalloc1(comm_size,&set_sizes);CHKERRQ(ierr);
  } else {
    set_sizes = NULL;
  }

  /* Gather number of keys from each rank */
  ierr = MPI_Gather(entries,1,MPI_INT,set_sizes,1,MPI_INT,0,comm);CHKERRMPI(ierr);
  if (comm_rank == 0) {
    ierr = PetscMalloc1(comm_size,&displace);CHKERRQ(ierr);
    displace[0] = 0;
    total = 0;
    for (ii=1; ii<comm_size; ii++) {
      total += set_sizes[ii - 1];
      displace[ii] = total;
    }
    total += set_sizes[comm_size - 1];
    ierr = PetscMalloc1(total,&recvset);CHKERRQ(ierr);
  } else {
    displace = NULL;
    recvset = NULL;
  }

  /* Gatherv keys from all ranks and intersect */
  ierr = MPI_Gatherv(set,*entries,MPI_INT,recvset,set_sizes,displace,MPI_INT,0,comm);CHKERRMPI(ierr);
  if (comm_rank == 0) {
    for (ii=1; ii<comm_size; ii++) {
      ierr = GarbageKeySortedIntersect_Private(set,entries,&recvset[displace[ii]],set_sizes[ii]);CHKERRQ(ierr);
    }
  }

  ierr = PetscFree(set_sizes);CHKERRQ(ierr);
  ierr = PetscFree(displace);CHKERRQ(ierr);
  ierr = PetscFree(recvset);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* Calculate block for given communicator and blocksize */
static PetscErrorCode CalculateBlock_Private(MPI_Comm comm,int blocksize)
{
  PetscErrorCode ierr;
  PetscInt       p,block,comm_size;

  PetscFunctionBegin;
  /* Calculate biggest power of blocksize smaller than communicator size */
  ierr = MPI_Comm_size(comm,&comm_size);CHKERRMPI(ierr);
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
  PetscErrorCode ierr;
  PetscInt       block,comm_rank,blockrank,intra_rank,leader;
  MPI_Comm       *intracom,*intercom;

  PetscFunctionBegin;
  /* Calculate biggest power of blocksize smaller than communicator size */
  block = CalculateBlock_Private(comm, blocksize);

  /* Create intra-communicators of size block or smaller */
  ierr = MPI_Comm_rank(comm,&comm_rank);CHKERRMPI(ierr);
  blockrank = comm_rank/block;
  ierr = PetscNew(&intracom);CHKERRQ(ierr);
  ierr = MPI_Comm_split(comm,blockrank,comm_rank,intracom);CHKERRMPI(ierr);
  ierr = MPI_Comm_set_attr(comm,Petsc_Garbage_IntraComm_keyval,(void*)intracom);CHKERRMPI(ierr);

  /* Create inter-communicators between rank 0 of all above comms */
  ierr = MPI_Comm_rank(*intracom,&intra_rank);CHKERRMPI(ierr);
  leader = (intra_rank == 0) ? 0 : MPI_UNDEFINED;
  ierr = PetscNew(&intercom);CHKERRQ(ierr);
  ierr = MPI_Comm_split(comm,leader,comm_rank,intercom);CHKERRMPI(ierr);
  ierr = MPI_Comm_set_attr(comm,Petsc_Garbage_InterComm_keyval,(void*)intercom);CHKERRMPI(ierr);
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
  PetscErrorCode ierr;
  PetscInt       ii,flag,entries,offset,intra_rank,inter_rank;
  PetscInt       *keys;
  PetscObject    *obj;
  PetscHMapObj   *garbage;
  MPI_Comm       *intracom,*intercom;

  PetscFunctionBegin;
  /* Get the garbage hash map */
  ierr = PetscCommDuplicate(comm,&comm,NULL);
  ierr = GarbageGetHMap_Private(comm,&garbage);

  /* Get the intra- and inter- communicators,if they exist,otherwise set them up */
  intracom = NULL;
  ierr = MPI_Comm_get_attr(comm,Petsc_Garbage_IntraComm_keyval,&intracom,&flag);CHKERRMPI(ierr);
  if (!flag) {
    ierr = GarbageSetupComms_Private(comm,blocksize);CHKERRQ(ierr);
    ierr = MPI_Comm_get_attr(comm,Petsc_Garbage_IntraComm_keyval,&intracom,&flag);CHKERRMPI(ierr);
  }
  ierr = MPI_Comm_get_attr(comm,Petsc_Garbage_InterComm_keyval,&intercom,&flag);CHKERRMPI(ierr);
  ierr = MPI_Comm_rank(*intracom,&intra_rank);CHKERRMPI(ierr);
  if (*intercom != MPI_COMM_NULL) {
    ierr = MPI_Comm_rank(*intercom,&inter_rank);CHKERRMPI(ierr);
  } else {
    inter_rank = MPI_UNDEFINED;
  }

  /* Get keys from garbage hash map and sort */
  ierr = PetscHMapObjGetSize(*garbage,&entries);CHKERRQ(ierr);
  ierr = PetscMalloc1(entries,&keys);CHKERRQ(ierr);
  offset = 0;
  ierr = PetscHMapObjGetKeys(*garbage,&offset,keys);CHKERRQ(ierr);
  ierr = PetscSortInt(entries,keys);CHKERRQ(ierr);

  /* Intracom gather and intersect */
  ierr = GarbageKeyGatherIntersect_Private(*intracom,keys,&entries);CHKERRQ(ierr);

  /* Intercom gather and intersect */
  if (*intercom != MPI_COMM_NULL) {
    ierr = GarbageKeyGatherIntersect_Private(*intercom,keys,&entries);CHKERRQ(ierr);
    /* Broadcast across intercom */
    ierr = MPI_Bcast(&entries,1,MPI_INT,0,*intercom);CHKERRMPI(ierr);
    ierr = MPI_Bcast(keys,entries,MPI_INT,0,*intercom);CHKERRMPI(ierr);
  }

  /* Broadcast across intracom */
  ierr = MPI_Bcast(&entries,1,MPI_INT,0,*intracom);CHKERRMPI(ierr);
  ierr = MPI_Bcast(keys,entries,MPI_INT,0,*intracom);CHKERRMPI(ierr);

  /* Collectively destroy objects objects that appear in garbage in
   * creation index order */
  for (ii = 0; ii < entries; ii++) {
    ierr = PetscHMapObjGet(*garbage,keys[ii],&obj);CHKERRQ(ierr);
    if (PetscCheckPointer((void*) obj,PETSC_OBJECT) && (obj != NULL)) {
      ierr = PetscObjectDestroy(obj);CHKERRQ(ierr);
      ierr = PetscFree(obj);CHKERRQ(ierr);
    }
    ierr = PetscHMapObjDel(*garbage,keys[ii]);CHKERRQ(ierr);
  }
  ierr = PetscFree(keys);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* Performs a collective intersection of one array per rank _recursively_
   Replaces above GatherIntersect function */
static PetscErrorCode GarbageKeyRecursiveGatherIntersect_Private(MPI_Comm comm,PetscInt *set,PetscInt *entries,PetscInt blocksize)
{
  PetscErrorCode ierr;
  PetscInt       flag,comm_size,intra_rank,inter_rank;
  MPI_Comm       *intracom,*intercom;

  PetscFunctionBegin;
  ierr = MPI_Comm_size(comm,&comm_size);CHKERRMPI(ierr);
  if (comm_size > blocksize) {
    /* Get the intra- and inter- communicators,if they exist,otherwise set them up */
    intracom = NULL;
    ierr = MPI_Comm_get_attr(comm,Petsc_Garbage_IntraComm_keyval,&intracom,&flag);CHKERRMPI(ierr);
    if (!flag) {
      ierr = GarbageSetupComms_Private(comm,blocksize);CHKERRQ(ierr);
      ierr = MPI_Comm_get_attr(comm,Petsc_Garbage_IntraComm_keyval,&intracom,&flag);CHKERRMPI(ierr);
    }
    ierr = MPI_Comm_get_attr(comm,Petsc_Garbage_InterComm_keyval,&intercom,&flag);CHKERRMPI(ierr);
    ierr = MPI_Comm_rank(*intracom,&intra_rank);CHKERRMPI(ierr);
    if (*intercom != MPI_COMM_NULL) {
      ierr = MPI_Comm_rank(*intercom,&inter_rank);CHKERRMPI(ierr);
    } else {
      inter_rank = MPI_UNDEFINED;
    }

    /* Gather and intersect on intracom recursively */
    ierr = GarbageKeyRecursiveGatherIntersect_Private(*intracom,set,entries,blocksize);CHKERRQ(ierr);
    /* Gather intersect over intercom */
    if (*intercom != MPI_COMM_NULL) {
      ierr = GarbageKeyGatherIntersect_Private(*intercom,set,entries);CHKERRQ(ierr);
    }
  } else {
    ierr = GarbageKeyGatherIntersect_Private(comm,set,entries);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/* Performs broadcast of resultant intersection _recursively_ */
static PetscErrorCode GarbageKeyRecursiveBcast_Private(MPI_Comm comm,PetscInt *set,PetscInt *entries)
{
  PetscInt ierr,flag;
  MPI_Comm *intracom,*intercom;

  PetscFunctionBegin;
  ierr = MPI_Comm_get_attr(comm,Petsc_Garbage_IntraComm_keyval,&intracom,&flag);CHKERRMPI(ierr);
  if (flag) {
    ierr = MPI_Comm_get_attr(comm,Petsc_Garbage_InterComm_keyval,&intercom,&flag);CHKERRMPI(ierr);
    if (*intercom != MPI_COMM_NULL) {
      /* Broadcast across intercom */
      ierr = MPI_Bcast(entries,1,MPI_INT,0,*intercom);CHKERRMPI(ierr);
      ierr = MPI_Bcast(set,*entries,MPI_INT,0,*intercom);CHKERRMPI(ierr);
    }
    /* Broadcast across intracom */
    ierr = GarbageKeyRecursiveBcast_Private(*intracom,set,entries);CHKERRQ(ierr);
  } else {
    /* If the comm has no intracom,we are at the recursion base case */
    ierr = MPI_Bcast(entries,1,MPI_INT,0,comm);CHKERRMPI(ierr);
    ierr = MPI_Bcast(set,*entries,MPI_INT,0,comm);CHKERRMPI(ierr);
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
  PetscInt     ii,ierr,entries,offset;
  PetscInt     *keys;
  PetscObject  *obj;
  PetscHMapObj *garbage;

  PetscFunctionBegin;
  /* Get keys from garbage hash map and sort */
  ierr = PetscCommDuplicate(comm,&comm,NULL);
  ierr = GarbageGetHMap_Private(comm,&garbage);
  ierr = PetscHMapObjGetSize(*garbage,&entries);CHKERRQ(ierr);
  ierr = PetscMalloc1(entries,&keys);CHKERRQ(ierr);
  offset = 0;
  ierr = PetscHMapObjGetKeys(*garbage,&offset,keys);CHKERRQ(ierr);

  ierr = PetscSortInt(entries,keys);CHKERRQ(ierr);

  /* Recursive gather+intersect and broadcast */
  ierr = GarbageKeyRecursiveGatherIntersect_Private(comm,keys,&entries,blocksize);CHKERRQ(ierr);
  ierr = GarbageKeyRecursiveBcast_Private(comm,keys,&entries);CHKERRQ(ierr);

  for (ii=0; ii<entries; ii++) {
    ierr = PetscHMapObjGet(*garbage,keys[ii],&obj);CHKERRQ(ierr);
    if (PetscCheckPointer((void*)obj,PETSC_OBJECT) && (obj != NULL)) {
      ierr = PetscObjectDestroy(obj);CHKERRQ(ierr);
      ierr = PetscFree(obj);CHKERRQ(ierr);
    }
    ierr = PetscHMapObjDel(*garbage,keys[ii]);CHKERRQ(ierr);
  }
  ierr = PetscFree(keys);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
