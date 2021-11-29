#include "petsc/private/garbagecollector.h"

/* MPI keyvals,only used internally */
PetscMPIInt Petsc_Garbage_HMap_keyval      = MPI_KEYVAL_INVALID;
PetscMPIInt Petsc_Garbage_IntraComm_keyval = MPI_KEYVAL_INVALID;
PetscMPIInt Petsc_Garbage_InterComm_keyval = MPI_KEYVAL_INVALID;

static PetscErrorCode GarbageKeyvalFree_Private(void);

/* Creates keyvals for storing attributes on MPI communicator */
static PetscErrorCode GarbageKeyvalCreate_Private(void)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (Petsc_Garbage_HMap_keyval == MPI_KEYVAL_INVALID) {
    ierr = PetscInfo(NULL,"Creating keyvals,should only happen once\n");CHKERRQ(ierr);
    ierr = MPI_Comm_create_keyval(MPI_COMM_NULL_COPY_FN,MPI_COMM_NULL_DELETE_FN,&Petsc_Garbage_HMap_keyval,(void*)0);CHKERRMPI(ierr);
    ierr = MPI_Comm_create_keyval(MPI_COMM_NULL_COPY_FN,MPI_COMM_NULL_DELETE_FN,&Petsc_Garbage_IntraComm_keyval,(void*)0);CHKERRMPI(ierr);
    ierr = MPI_Comm_create_keyval(MPI_COMM_NULL_COPY_FN,MPI_COMM_NULL_DELETE_FN,&Petsc_Garbage_InterComm_keyval,(void*)0);CHKERRMPI(ierr);
    ierr = PetscRegisterFinalize(GarbageKeyvalFree_Private);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/* Destroys keyvals */
static PetscErrorCode GarbageKeyvalFree_Private(void)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscInfo(NULL,"Freeing Petsc_Garbage_HMap_keyval,Petsc_Garbage_IntraComm_keyval and Petsc_Garbage_InterComm_keyval keyvals\n");CHKERRQ(ierr);
  ierr = MPI_Comm_free_keyval(&Petsc_Garbage_HMap_keyval);CHKERRMPI(ierr);
  ierr = MPI_Comm_free_keyval(&Petsc_Garbage_IntraComm_keyval);CHKERRMPI(ierr);
  ierr = MPI_Comm_free_keyval(&Petsc_Garbage_InterComm_keyval);CHKERRMPI(ierr);
  PetscFunctionReturn(0);
}

/* Fetches garbage hashmap from communicator */
static PetscErrorCode GarbageGetHMap_Private(MPI_Comm comm,PetscHMapObj **garbage)
{
  void           *get_tmp;
  PetscInt       flag;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MPI_Comm_get_attr(comm,Petsc_Garbage_HMap_keyval,&get_tmp,&flag);CHKERRMPI(ierr);
  if (!flag) {
    /* No garbage,create one */
    ierr = PetscMalloc1(1,*garbage);CHKERRQ(ierr);
    ierr = PetscHMapObjCreate(*garbage);CHKERRQ(ierr);
    ierr = MPI_Comm_set_attr(comm,Petsc_Garbage_HMap_keyval,*garbage);CHKERRMPI(ierr);
  } else {
    *garbage = (PetscHMapObj*)get_tmp;
  }
  PetscFunctionReturn(0);
}

/* Analogue to PetscObjectDestroy for use in managed languages */
PetscErrorCode PetscObjectDelayedDestroy(PetscObject *obj)
{
  MPI_Comm       petsc_comm;
  PetscErrorCode ierr;
  PetscObject    *duplicate;
  PetscHMapObj   *garbage;

  PetscFunctionBegin;
  /* Don't stash NULL pointers */
  if (*obj != NULL) {
    /* If the keyvals aren't yet set up,create them and register finalizer */
    ierr = GarbageKeyvalCreate_Private();CHKERRQ(ierr);
    ierr = PetscObjectGetComm(*obj,&petsc_comm);CHKERRQ(ierr);
    ierr = GarbageGetHMap_Private(petsc_comm,&garbage);CHKERRQ(ierr);
    /* TODO: Does there need to be a check here for a NULL pointer? */

    /* Duplicate object header so managed language can clean up original */
    ierr = PetscMalloc1(1,&duplicate);CHKERRQ(ierr);
    ierr = PetscMemcpy(duplicate,obj,sizeof(PetscObject));CHKERRQ(ierr);
    ierr = PetscHMapObjSet(*garbage,(*duplicate)->cidx,duplicate);CHKERRQ(ierr);
    *obj = NULL;
  } /* TODO: Is an else clause necessary? */
  PetscFunctionReturn(0);
}

/* Performs the intersection of 2 sorted arrays seta and setb of lengths
 * lena and lenb respectively,returning the result in seta and len a
 * This is then an O(n) operation */
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

  /* Free memory */
  ierr = PetscFree(set_sizes);CHKERRQ(ierr);
  ierr = PetscFree(displace);CHKERRQ(ierr);
  ierr = PetscFree(recvset);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* Setup inter- and intra- communicators for more efficient MPI */
static PetscErrorCode GarbageSetupComms_Private(MPI_Comm comm,int blocksize)
{
  PetscErrorCode ierr;
  PetscInt       p,block,comm_size,comm_rank,blockrank,intra_rank,leader;
  MPI_Comm       *intracom,*intercom;

  PetscFunctionBegin;
  /* Calculate biggest power of blocksize smaller than communicator size */
  ierr = MPI_Comm_size(comm,&comm_size);CHKERRMPI(ierr);
  p = (int)PetscFloorReal(PetscLogReal((double)comm_size)/PetscLogReal((double)blocksize));
  block = (int)PetscPowReal((double)blocksize,(double)p);
  if ((block == comm_size) && (comm_size != 1)) {
    block = (int)PetscPowReal((double)blocksize,(double)(p - 1));
  }

  /* Create intra-communicators of size block or smaller */
  ierr = MPI_Comm_rank(comm,&comm_rank);CHKERRMPI(ierr);
  blockrank = comm_rank/block;
  intracom = (MPI_Comm*)malloc(sizeof(MPI_Comm));
  ierr = MPI_Comm_split(comm,blockrank,comm_rank,intracom);CHKERRMPI(ierr);
  ierr = MPI_Comm_set_attr(comm,Petsc_Garbage_IntraComm_keyval,(void*)intracom);CHKERRMPI(ierr);

  /* Create inter-communicators between rank 0 of all above comms */
  ierr = MPI_Comm_rank(*intracom,&intra_rank);CHKERRMPI(ierr);
  leader = (intra_rank == 0) ? 0 : MPI_UNDEFINED;
  intercom = (MPI_Comm*)malloc(sizeof(MPI_Comm));
  ierr = MPI_Comm_split(comm,leader,comm_rank,intercom);CHKERRMPI(ierr);
  ierr = MPI_Comm_set_attr(comm,Petsc_Garbage_InterComm_keyval,(void*)intercom);CHKERRMPI(ierr);
  PetscFunctionReturn(0);
}

/* Implements garbage collection for objects destroyed using DelayedObjectDestroy */
PetscErrorCode PetscGarbageCleanup(MPI_Comm comm,PetscInt blocksize)
{
  PetscErrorCode ierr;
  PetscInt       ii,flag,entries,offset,intra_rank,inter_rank;
  PetscInt       *keys;
  PetscObject    *obj;
  PetscHMapObj   *garbage;
  MPI_Comm       *intracom,*intercom;

  PetscFunctionBegin;
  /* If the keyvals aren't yet set up,create them and register finalizer */
  ierr = GarbageKeyvalCreate_Private();CHKERRQ(ierr);
  ierr = PetscCommDuplicate(comm,&comm,NULL);

  /* Get the garbage hash map */
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
    } /* TODO: else? */
    ierr = PetscHMapObjDel(*garbage,keys[ii]);CHKERRQ(ierr);
  }
  PetscFree(keys);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* Performs a collective intersection of one array per rank _recursively_
 * Replaces above GatherIntersect function */
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

/* Implements garbage collection for objects destroyed using DelayedObjectDestroy _recursively_
 * Replaces above PetscGarbageCleanup function */
PetscErrorCode PetscRecursiveGarbageCleanup(MPI_Comm comm,PetscInt blocksize)
{
  PetscInt     ii,ierr,entries,offset;
  PetscInt     *keys;
  PetscObject  *obj;
  PetscHMapObj *garbage;

  PetscFunctionBegin;
  /* If the keyvals aren't yet set up,create them and register finalizer */
  ierr = GarbageKeyvalCreate_Private();CHKERRQ(ierr);
  ierr = PetscCommDuplicate(comm,&comm,NULL);

  /* Get keys from garbage hash map and sort */
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
    } /* TODO: else? */
    ierr = PetscHMapObjDel(*garbage,keys[ii]);CHKERRQ(ierr);
  }
  PetscFree(keys);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
