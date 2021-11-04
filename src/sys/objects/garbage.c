#include "petsc/private/garbagecollector.h"

PetscMPIInt GARBAGE = MPI_KEYVAL_INVALID;
PetscMPIInt INTRA = MPI_KEYVAL_INVALID;
PetscMPIInt INTER = MPI_KEYVAL_INVALID;

PetscErrorCode keyval_free(void);

PetscErrorCode keyval_create(void){
  PetscErrorCode ierr;

  if(GARBAGE == MPI_KEYVAL_INVALID){
    ierr = PetscPrintf(PETSC_COMM_WORLD, "Creating keyvals, should only happen once\n"); CHKERRQ(ierr);
    ierr = MPI_Comm_create_keyval(MPI_COMM_NULL_COPY_FN, MPI_COMM_NULL_DELETE_FN, &GARBAGE, (void*)0); CHKERRMPI(ierr);
    ierr = MPI_Comm_create_keyval(MPI_COMM_NULL_COPY_FN, MPI_COMM_NULL_DELETE_FN, &INTRA, (void*)0); CHKERRMPI(ierr);
    ierr = MPI_Comm_create_keyval(MPI_COMM_NULL_COPY_FN, MPI_COMM_NULL_DELETE_FN, &INTER, (void*)0); CHKERRMPI(ierr);
    ierr = PetscRegisterFinalize(keyval_free); CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

PetscErrorCode keyval_free(void){
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscInfo(NULL,"Freeing CREATION_IDX, GARBAGE, INTRA and INTER keyvals\n"); CHKERRQ(ierr);
  ierr = MPI_Comm_free_keyval(&GARBAGE); CHKERRMPI(ierr);
  ierr = MPI_Comm_free_keyval(&INTRA); CHKERRMPI(ierr);
  ierr = MPI_Comm_free_keyval(&INTER); CHKERRMPI(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode getGarbage(MPI_Comm comm, PetscHMapObj **garbage){
  void *get_tmp;
  PetscInt flag;
  PetscErrorCode ierr;
  ierr = MPI_Comm_get_attr(comm, GARBAGE, &get_tmp, &flag); CHKERRMPI(ierr);
  if(!flag){
    /* No garbage, create one */
    printf("\tCreating garbage\n");
    *garbage =(PetscHMapObj*) malloc(sizeof(PetscHMapObj));
    ierr = PetscHMapObjCreate(*garbage); CHKERRQ(ierr);
    ierr = MPI_Comm_set_attr(comm, GARBAGE, *garbage); CHKERRMPI(ierr);
  }else{;
    *garbage = (PetscHMapObj *) get_tmp;
  }
  PetscFunctionReturn(0);
}

PetscErrorCode DelayedObjectDestroy(PetscObject *obj){
  MPI_Comm petsc_comm;
  PetscErrorCode ierr;
  PetscObject *duplicate;
  PetscHMapObj *garbage;

  /* Don't stash NULL pointers */
  if (*obj != NULL){
    /* Do not use PetscPrintf in here */
    //~ printf("Enter destroy function\n");
    /* If the keyvals aren't yet set up, create them and register finalizer */
    ierr = keyval_create(); CHKERRQ(ierr);

    //~ obj = (PetscObject) *vec;
    ierr = PetscObjectGetComm(*obj, &petsc_comm); CHKERRQ(ierr);
    //~ printf("A %i\n", (int)petsc_comm);
    ierr = getGarbage(petsc_comm, &garbage); CHKERRQ(ierr);
    if(garbage == NULL){
      printf("\tgarbage is null\n");
    }

    /* Duplicate object header so interpreted language can clean up original */
    ierr = PetscMalloc1(1, &duplicate); CHKERRQ(ierr);
    ierr = PetscMemcpy(duplicate, obj, sizeof(PetscObject)); CHKERRQ(ierr);
    //~ printf("\t garbage pointer: %p\n", (void *)garbage);
    //~ printf("\t creation index: %i\n", obj->cidx);
    ierr = PetscHMapObjSet(*garbage, (*duplicate)->cidx, duplicate); CHKERRQ(ierr);
    printf("Stashed %p of type *%s on comm %i\n", duplicate, (*duplicate)->class_name, (int)petsc_comm);
    printf("Originally: %p of type *%s\n", obj, (*obj)->class_name);
    //~ *obj = NULL;
  } else {
    printf("Attempt to stash NULL pointer\n");
  }
  //~ ierr = PetscPrintf(petsc_comm, "Exit destroy function\n"); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

void sorted_intersect(int *seta, int *lena, int *setb, int lenb){
  /* The arrays seta and setb must be sorted */
  int ii, counter = 0;
  int *endb;
  endb = setb + lenb;

  for(ii = 0; ii < *lena; ii++){
    //~ printf("A: %i \n", seta[ii]);
    while((seta[ii] > *setb) && (setb < endb)){
      //~ printf("B: %i \n", *setb);
      setb++;
    }
    if(seta[ii] == *setb){
      //~ printf("SAME: %i\n", seta[ii]);
      seta[counter] = seta[ii];
      counter++;
    }
  }
  //~ printf("\n");
  for(ii = 0; ii < counter; ii++){
    //~ printf("%i ", seta[ii]);
  }
  //~ printf("\n");
  *lena = counter;
}

PetscErrorCode gather_intersect(MPI_Comm comm, PetscInt *set, PetscInt *entries){
  PetscInt ii, comm_size, comm_rank, total;
  PetscInt *set_sizes, *displace, *recvset;
  PetscErrorCode ierr;

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
      displace[ii] = set_sizes[ii - 1];
      total += set_sizes[ii - 1];
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

PetscErrorCode PetscGarbageCleanup(MPI_Comm comm, PetscInt blocksize){
  PetscInt ii, ierr, flag, entries, offset;
  PetscInt comm_rank, blockrank;
  PetscInt intra_rank, leader;
  PetscInt inter_rank;
  PetscInt *keys;
  PetscObject *obj;
  PetscHMapObj *garbage;
  MPI_Comm *intracom, *intercom;

  /* If the keyvals aren't yet set up, create them and register finalizer */
  ierr = keyval_create(); CHKERRQ(ierr);

  ierr = PetscCommDuplicate(comm, &comm, NULL);
  ierr = PetscPrintf(comm, "Enter cleanup function\n"); CHKERRQ(ierr);
  printf("Cleaning up on comm: %i\n", (int) comm);

  /* Get the garbage hash map */
  ierr = getGarbage(comm, &garbage);

  /* Get the intra- and inter- communicators, if they exist, otherwise set them up */
  ierr = MPI_Comm_rank(comm, &comm_rank); CHKERRMPI(ierr);
  blockrank = comm_rank/blocksize;
  intracom = NULL;
  ierr = MPI_Comm_get_attr(comm, INTRA, &intracom, &flag); CHKERRMPI(ierr);
  if(!flag){
    intracom = (MPI_Comm*)malloc(sizeof(MPI_Comm));
    ierr = MPI_Comm_split(comm, blockrank, comm_rank, intracom);
    ierr = MPI_Comm_set_attr(comm, INTRA, (void*)intracom); CHKERRMPI(ierr);
  }
  ierr = MPI_Comm_rank(*intracom, &intra_rank); CHKERRMPI(ierr);

  if(intra_rank == 0){
    leader = 0;
  }else{
    leader = MPI_UNDEFINED;
  }

  intercom = NULL;
  ierr = MPI_Comm_get_attr(comm, INTER, &intercom, &flag); CHKERRMPI(ierr);
  if(!flag){
    intercom = (MPI_Comm*)malloc(sizeof(MPI_Comm));
    ierr = MPI_Comm_split(comm, leader, comm_rank, intercom);
    ierr = MPI_Comm_set_attr(comm, INTER, (void*)intercom); CHKERRMPI(ierr);
  }
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

  printf("\tB%iR%i: Keys = [", blockrank, intra_rank);
  for(ii = 0; ii < entries; ii++){
    printf("%i ", keys[ii]);
  }
  printf("]");

  //~ ierr = PetscPrintf(comm, "SORTED KEYS:\n"); CHKERRQ(ierr);
  ierr = PetscSortInt(entries, keys); CHKERRQ(ierr);
  //~ for(ix = 0; ix < entries; ix++){
    //~ ierr = PetscPrintf(comm, "Key %i: %i\n", ix, keys[ix]); CHKERRQ(ierr);
  //~ }

  printf(" --> [");
  for(ii = 0; ii < entries; ii++){
    printf("%i ", keys[ii]);
  }
  printf("]\n");

  /* Intracom gather and intersect */
  ierr = gather_intersect(*intracom, keys, &entries); CHKERRQ(ierr);
  if(intra_rank == 0){
    printf("\tB%iR%i: intersection = [", blockrank, intra_rank);
    for(ii = 0; ii < entries; ii++){
      printf("%i ", keys[ii]);
    }
    printf("]\n");
  }

  /* Intercom gather and intersect */
  if(*intercom != MPI_COMM_NULL){
    ierr = gather_intersect(*intercom, keys, &entries); CHKERRQ(ierr);
    if(inter_rank == 0){
      printf("\tB%iR%i: intersection2 = [", blockrank, intra_rank);
      for(ii = 0; ii < entries; ii++){
        printf("%i ", keys[ii]);
      }
      printf("]\n");
    }
    /* Broadcast across intercom */
    ierr = MPI_Bcast(&entries, 1, MPI_INT, 0, *intercom); CHKERRMPI(ierr);
    ierr = MPI_Bcast(keys, entries, MPI_INT, 0, *intercom); CHKERRMPI(ierr);
  }

  /* Broadcast across intracom */
  ierr = MPI_Bcast(&entries, 1, MPI_INT, 0, *intracom); CHKERRMPI(ierr);
  ierr = MPI_Bcast(keys, entries, MPI_INT, 0, *intracom); CHKERRMPI(ierr);

  printf("\tB%iR%i: intersection3 = [", blockrank, intra_rank);
  for(ii = 0; ii < entries; ii++){
    printf("%i ", keys[ii]);
  }
  printf("]\n");

  for(ii = 0; ii < entries; ii++){
    ierr = PetscHMapObjGet(*garbage, keys[ii], &obj); CHKERRQ(ierr);
    printf("Destroying pointer %p\n", obj);
    if(PetscCheckPointer((void*) obj, PETSC_OBJECT) && (obj != NULL)){
      printf("Value %p of type %s on comm %i\n", *obj, (*obj)->class_name,(int)comm);
      ierr = PetscObjectDestroy(obj); CHKERRQ(ierr);
      ierr = PetscFree(obj); CHKERRQ(ierr);
    }else{
      printf("\tpointer not to valid object\n");
    }
    ierr = PetscHMapObjDel(*garbage, keys[ii]); CHKERRQ(ierr);
  }

  PetscFree(keys); CHKERRQ(ierr);

  ierr = PetscPrintf(comm, "Exit cleanup function\n"); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

