#include "petsc/private/garbagecollector.h"

/* MPI keyvals, only used internally */
PetscMPIInt GARBAGE = MPI_KEYVAL_INVALID;
PetscMPIInt INTRA = MPI_KEYVAL_INVALID;
PetscMPIInt INTER = MPI_KEYVAL_INVALID;

PetscErrorCode keyval_free(void);

/* Creates keyvals for storing attributes on MPI communicator */
PetscErrorCode keyval_create(void){
  PetscErrorCode ierr;
  if(GARBAGE == MPI_KEYVAL_INVALID){
    ierr = PetscInfo(NULL,"Creating keyvals, should only happen once\n"); CHKERRQ(ierr);
    ierr = MPI_Comm_create_keyval(MPI_COMM_NULL_COPY_FN, MPI_COMM_NULL_DELETE_FN, &GARBAGE, (void*)0); CHKERRMPI(ierr);
    ierr = MPI_Comm_create_keyval(MPI_COMM_NULL_COPY_FN, MPI_COMM_NULL_DELETE_FN, &INTRA, (void*)0); CHKERRMPI(ierr);
    ierr = MPI_Comm_create_keyval(MPI_COMM_NULL_COPY_FN, MPI_COMM_NULL_DELETE_FN, &INTER, (void*)0); CHKERRMPI(ierr);
    ierr = PetscRegisterFinalize(keyval_free); CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/* Destroys keyvals */
PetscErrorCode keyval_free(void){
  PetscErrorCode ierr;
  PetscFunctionBegin;
  ierr = PetscInfo(NULL,"Freeing CREATION_IDX, GARBAGE, INTRA and INTER keyvals\n"); CHKERRQ(ierr);
  ierr = MPI_Comm_free_keyval(&GARBAGE); CHKERRMPI(ierr);
  ierr = MPI_Comm_free_keyval(&INTRA); CHKERRMPI(ierr);
  ierr = MPI_Comm_free_keyval(&INTER); CHKERRMPI(ierr);
  PetscFunctionReturn(0);
}

/* Fetches garbage hashmap from communicator */
PetscErrorCode getGarbage(MPI_Comm comm, PetscHMapObj **garbage){
  void *get_tmp;
  PetscInt flag;
  PetscErrorCode ierr;
  ierr = MPI_Comm_get_attr(comm, GARBAGE, &get_tmp, &flag); CHKERRMPI(ierr);
  if(!flag){
    /* No garbage, create one */
    *garbage =(PetscHMapObj*) malloc(sizeof(PetscHMapObj));
    ierr = PetscHMapObjCreate(*garbage); CHKERRQ(ierr);
    ierr = MPI_Comm_set_attr(comm, GARBAGE, *garbage); CHKERRMPI(ierr);
  }else{;
    *garbage = (PetscHMapObj *) get_tmp;
  }
  PetscFunctionReturn(0);
}

/* Analogue to PetscObjectDestroy for use in managed languages */
PetscErrorCode DelayedObjectDestroy(PetscObject *obj){
  MPI_Comm petsc_comm;
  PetscErrorCode ierr;
  PetscObject *duplicate;
  PetscHMapObj *garbage;

  /* Don't stash NULL pointers */
  if (*obj != NULL){
    /* If the keyvals aren't yet set up, create them and register finalizer */
    ierr = keyval_create(); CHKERRQ(ierr);
    ierr = PetscObjectGetComm(*obj, &petsc_comm); CHKERRQ(ierr);
    ierr = getGarbage(petsc_comm, &garbage); CHKERRQ(ierr);
    /* TODO: Does there need to be a check here for a NULL pointer? */

    /* Duplicate object header so managed language can clean up original */
    ierr = PetscMalloc1(1, &duplicate); CHKERRQ(ierr);
    ierr = PetscMemcpy(duplicate, obj, sizeof(PetscObject)); CHKERRQ(ierr);
    ierr = PetscHMapObjSet(*garbage, (*duplicate)->cidx, duplicate); CHKERRQ(ierr);
    *obj = NULL;
  } /* TODO: Is an else clause necessary? */
  PetscFunctionReturn(0);
}

/* Performs the intersection of 2 sorted arrays seta and setb of lengths
 * lena and lenb respectively, returning the result in seta and len a
 * This is then an O(n) operation */
void sorted_intersect(int *seta, int *lena, int *setb, int lenb){
  /* The arrays seta and setb MUST be sorted! */
  int ii, counter = 0;
  int *endb;
  endb = setb + lenb;

  for(ii = 0; ii < *lena; ii++){
    while((seta[ii] > *setb) && (setb < endb)){
      setb++;
    }
    if(seta[ii] == *setb){
      seta[counter] = seta[ii];
      counter++;
    }
  }
  *lena = counter;
}

/* Performs a collective intersection of one array per rank */
PetscErrorCode gather_intersect(MPI_Comm comm, PetscInt *set, PetscInt *entries){
  PetscErrorCode ierr;
  PetscInt ii, comm_size, comm_rank, total;
  PetscInt *set_sizes, *displace, *recvset;

  /* Gather and intersect on comm */
  ierr = MPI_Comm_size(comm, &comm_size); CHKERRMPI(ierr);
  ierr = MPI_Comm_rank(comm, &comm_rank); CHKERRMPI(ierr);
  if(comm_rank == 0){
    ierr = PetscMalloc1(comm_size, &set_sizes); CHKERRQ(ierr);
  }else{
    set_sizes = NULL;
  }

  /* Gather number of keys from each rank */
  ierr = MPI_Gather(entries, 1, MPI_INT, set_sizes, 1, MPI_INT, 0, comm); CHKERRMPI(ierr);
  if(comm_rank == 0){
    ierr = PetscMalloc1(comm_size, &displace); CHKERRQ(ierr);
    displace[0] = 0;
    total = 0;
    for(ii = 1; ii < comm_size; ii++){
      total += set_sizes[ii - 1];
      displace[ii] = total;
    }
    total += set_sizes[comm_size - 1];
    ierr = PetscMalloc1(total, &recvset); CHKERRQ(ierr);
  }else{
    displace = NULL;
    recvset = NULL;
  }

  /* Gatherv keys from all ranks and intersect */
  ierr = MPI_Gatherv(set, *entries, MPI_INT, recvset, set_sizes, displace, MPI_INT, 0, comm); CHKERRMPI(ierr);
  if(comm_rank == 0){
    for(ii = 1; ii < comm_size; ii++){
      sorted_intersect(set, entries, &recvset[displace[ii]], set_sizes[ii]);
    }
  }

  /* Free memory */
  ierr = PetscFree(set_sizes); CHKERRQ(ierr);
  ierr = PetscFree(displace); CHKERRQ(ierr);
  ierr = PetscFree(recvset); CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

/* Setup inter- and intra- communicators for more efficient MPI */
PetscErrorCode SetupComms(MPI_Comm comm, int blocksize){
  PetscErrorCode ierr;
  PetscInt p, block;
  PetscInt comm_size, comm_rank, blockrank;
  PetscInt intra_rank, leader;
  MPI_Comm *intracom, *intercom;

  /* Calculate biggest power of blocksize smaller than communicator size */
  ierr = MPI_Comm_size(comm, &comm_size); CHKERRMPI(ierr);
  p = (int)floor(log((double)comm_size)/log((double)blocksize));
  block = (int)pow((double)blocksize, (double)p);
  if((block == comm_size) && (comm_size != 1)){
    block = (int)pow((double)blocksize, (double)(p - 1));
  }

  /* Create intra-communicators of size block or smaller */
  ierr = MPI_Comm_rank(comm, &comm_rank); CHKERRMPI(ierr);
  blockrank = comm_rank/block;
  intracom = (MPI_Comm*)malloc(sizeof(MPI_Comm));
  ierr = MPI_Comm_split(comm, blockrank, comm_rank, intracom);
  ierr = MPI_Comm_set_attr(comm, INTRA, (void*)intracom); CHKERRMPI(ierr);

  /* Create inter-communicators between rank 0 of all above comms */
  ierr = MPI_Comm_rank(*intracom, &intra_rank); CHKERRMPI(ierr);
  leader = (intra_rank == 0) ? 0 : MPI_UNDEFINED;
  intercom = (MPI_Comm*)malloc(sizeof(MPI_Comm));
  ierr = MPI_Comm_split(comm, leader, comm_rank, intercom);
  ierr = MPI_Comm_set_attr(comm, INTER, (void*)intercom); CHKERRMPI(ierr);

  PetscFunctionReturn(0);
}

/* Implements garbage collection for objects destroyed using DelayedObjectDestroy */
PetscErrorCode PetscGarbageCleanup(MPI_Comm comm, PetscInt blocksize){
  PetscErrorCode ierr;
  PetscInt ii, flag, entries, offset;
  PetscInt intra_rank, inter_rank;
  PetscInt *keys;
  PetscObject *obj;
  PetscHMapObj *garbage;
  MPI_Comm *intracom, *intercom;

  /* If the keyvals aren't yet set up, create them and register finalizer */
  ierr = keyval_create(); CHKERRQ(ierr);
  ierr = PetscCommDuplicate(comm, &comm, NULL);

  /* Get the garbage hash map */
  ierr = getGarbage(comm, &garbage);

  /* Get the intra- and inter- communicators, if they exist, otherwise set them up */
  intracom = NULL;
  ierr = MPI_Comm_get_attr(comm, INTRA, &intracom, &flag); CHKERRMPI(ierr);
  if(!flag){
    ierr = SetupComms(comm, blocksize); CHKERRQ(ierr);
    ierr = MPI_Comm_get_attr(comm, INTRA, &intracom, &flag); CHKERRMPI(ierr);
  }
  ierr = MPI_Comm_get_attr(comm, INTER, &intercom, &flag); CHKERRMPI(ierr);
  ierr = MPI_Comm_rank(*intracom, &intra_rank); CHKERRMPI(ierr);
  if(*intercom != MPI_COMM_NULL){
    ierr = MPI_Comm_rank(*intercom, &inter_rank); CHKERRMPI(ierr);
  }else{
    inter_rank = MPI_UNDEFINED;
  }

  /* Get keys from garbage hash map and sort */
  ierr = PetscHMapObjGetSize(*garbage, &entries); CHKERRQ(ierr);
  ierr = PetscMalloc1(entries, &keys); CHKERRQ(ierr);
  offset = 0;
  ierr = PetscHMapObjGetKeys(*garbage, &offset, keys); CHKERRQ(ierr);
  ierr = PetscSortInt(entries, keys); CHKERRQ(ierr);

  /* Intracom gather and intersect */
  ierr = gather_intersect(*intracom, keys, &entries); CHKERRQ(ierr);

  /* Intercom gather and intersect */
  if(*intercom != MPI_COMM_NULL){
    ierr = gather_intersect(*intercom, keys, &entries); CHKERRQ(ierr);
    /* Broadcast across intercom */
    ierr = MPI_Bcast(&entries, 1, MPI_INT, 0, *intercom); CHKERRMPI(ierr);
    ierr = MPI_Bcast(keys, entries, MPI_INT, 0, *intercom); CHKERRMPI(ierr);
  }

  /* Broadcast across intracom */
  ierr = MPI_Bcast(&entries, 1, MPI_INT, 0, *intracom); CHKERRMPI(ierr);
  ierr = MPI_Bcast(keys, entries, MPI_INT, 0, *intracom); CHKERRMPI(ierr);

  /* Collectively destroy objects objects that appear in garbage in
   * creation index order */
  for(ii = 0; ii < entries; ii++){
    ierr = PetscHMapObjGet(*garbage, keys[ii], &obj); CHKERRQ(ierr);
    if(PetscCheckPointer((void*) obj, PETSC_OBJECT) && (obj != NULL)){
      ierr = PetscObjectDestroy(obj); CHKERRQ(ierr);
      ierr = PetscFree(obj); CHKERRQ(ierr);
    } /* TODO: else? */
    ierr = PetscHMapObjDel(*garbage, keys[ii]); CHKERRQ(ierr);
  }

  PetscFree(keys); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* Performs a collective intersection of one array per rank _recursively_
 * Replaces above GatherIntersect function */
PetscErrorCode RecursiveGatherIntersect(MPI_Comm comm, PetscInt *set, PetscInt *entries, PetscInt blocksize){
  PetscErrorCode ierr;
  PetscInt flag, comm_size;
  PetscInt intra_rank, inter_rank;
  MPI_Comm *intracom, *intercom;

  ierr = MPI_Comm_size(comm, &comm_size); CHKERRMPI(ierr);
  if(comm_size > blocksize){
    /* Get the intra- and inter- communicators, if they exist, otherwise set them up */
    intracom = NULL;
    ierr = MPI_Comm_get_attr(comm, INTRA, &intracom, &flag); CHKERRMPI(ierr);
    if(!flag){
      ierr = SetupComms(comm, blocksize); CHKERRQ(ierr);
      ierr = MPI_Comm_get_attr(comm, INTRA, &intracom, &flag); CHKERRMPI(ierr);
    }
    ierr = MPI_Comm_get_attr(comm, INTER, &intercom, &flag); CHKERRMPI(ierr);
    ierr = MPI_Comm_rank(*intracom, &intra_rank); CHKERRMPI(ierr);
    if(*intercom != MPI_COMM_NULL){
      ierr = MPI_Comm_rank(*intercom, &inter_rank); CHKERRMPI(ierr);
    }else{
      inter_rank = MPI_UNDEFINED;
    }

    /* Gather and intersect on intracom recursively */
    ierr = RecursiveGatherIntersect(*intracom, set, entries, blocksize); CHKERRQ(ierr);
    /* Gather intersect over intercom */
    if(*intercom != MPI_COMM_NULL){
      ierr = gather_intersect(*intercom, set, entries); CHKERRQ(ierr);
    }
  }else{
    ierr = gather_intersect(comm, set, entries); CHKERRQ(ierr);
  }

  PetscFunctionReturn(0);
}

/* Performs broadcast of resultant intersection _recursively_ */
PetscErrorCode RecursiveBcast(MPI_Comm comm, PetscInt *set, PetscInt *entries){
  PetscInt ierr, flag;
  MPI_Comm *intracom, *intercom;

  ierr = MPI_Comm_get_attr(comm, INTRA, &intracom, &flag); CHKERRMPI(ierr);
  if(flag){
    ierr = MPI_Comm_get_attr(comm, INTER, &intercom, &flag); CHKERRMPI(ierr);
    if(*intercom != MPI_COMM_NULL){
      /* Broadcast across intercom */
      ierr = MPI_Bcast(entries, 1, MPI_INT, 0, *intercom); CHKERRMPI(ierr);
      ierr = MPI_Bcast(set, *entries, MPI_INT, 0, *intercom); CHKERRMPI(ierr);
    }
    /* Broadcast across intracom */
    ierr = RecursiveBcast(*intracom, set, entries); CHKERRQ(ierr);
  }else{
    /* If the comm has no intracom, we are at the recursion base case */
    ierr = MPI_Bcast(entries, 1, MPI_INT, 0, comm); CHKERRMPI(ierr);
    ierr = MPI_Bcast(set, *entries, MPI_INT, 0, comm); CHKERRMPI(ierr);
  }

  PetscFunctionReturn(0);
}

/* Implements garbage collection for objects destroyed using DelayedObjectDestroy _recursively_
 * Replaces above PetscGarbageCleanup function */
PetscErrorCode PetscRecursiveGarbageCleanup(MPI_Comm comm, PetscInt blocksize){
  PetscInt ii, ierr, entries, offset;
  PetscInt *keys;
  PetscObject *obj;
  PetscHMapObj *garbage;

  /* If the keyvals aren't yet set up, create them and register finalizer */
  ierr = keyval_create(); CHKERRQ(ierr);
  ierr = PetscCommDuplicate(comm, &comm, NULL);

  /* Get keys from garbage hash map and sort */
  ierr = getGarbage(comm, &garbage);
  ierr = PetscHMapObjGetSize(*garbage, &entries); CHKERRQ(ierr);
  ierr = PetscMalloc1(entries, &keys); CHKERRQ(ierr);
  offset = 0;
  ierr = PetscHMapObjGetKeys(*garbage, &offset, keys); CHKERRQ(ierr);

  PetscInt comm_rank, blockrank; MPI_Comm_rank(comm, &comm_rank); blockrank = comm_rank/blocksize;

  ierr = PetscSortInt(entries, keys); CHKERRQ(ierr);

  /* Recursive gather+intersect and broadcast */
  ierr = RecursiveGatherIntersect(comm, keys, &entries, blocksize); CHKERRQ(ierr);
  ierr = RecursiveBcast(comm, keys, &entries); CHKERRQ(ierr);

  for(ii = 0; ii < entries; ii++){
    ierr = PetscHMapObjGet(*garbage, keys[ii], &obj); CHKERRQ(ierr);
    if(PetscCheckPointer((void*) obj, PETSC_OBJECT) && (obj != NULL)){
      ierr = PetscObjectDestroy(obj); CHKERRQ(ierr);
      ierr = PetscFree(obj); CHKERRQ(ierr);
    } /* TODO: else? */
    ierr = PetscHMapObjDel(*garbage, keys[ii]); CHKERRQ(ierr);
  }

  PetscFree(keys); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
