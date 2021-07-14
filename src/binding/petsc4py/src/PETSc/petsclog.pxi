cdef extern from * nogil:

    ctypedef double PetscLogDouble
    ctypedef struct PetscEventPerfInfo:
        int count
        PetscLogDouble flops, time, bytes
        PetscLogDouble numMessages
        PetscLogDouble messageLength
        PetscLogDouble numReductions

    ctypedef struct PetscEventRegInfo:
        char *name
    int PetscLogDefaultBegin()
    int PetscLogAllBegin()
    int PetscLogView(PetscViewer)

    int PetscLogFlops(PetscLogDouble)
    int PetscGetFlops(PetscLogDouble*)
    int PetscLogBytes(PetscLogDouble)
    int PetscGetBytes(PetscLogDouble*)
    int PetscGetCPUTime(PetscLogDouble*)
    int PetscMallocGetCurrentUsage(PetscLogDouble*)
    int PetscMemoryGetCurrentUsage(PetscLogDouble*)

    int PetscTime(PetscLogDouble*)
    int PetscTimeSubtract(PetscLogDouble*)
    int PetscTimeAdd(PetscLogDouble*)

    struct _n_PetscEventPerfLog:
        int numEvents
        PetscEventPerfInfo* eventInfo
    ctypedef _n_PetscEventPerfLog* PetscEventPerfLog
    struct _n_PetscEventRegLog:
        int numEvents
        PetscEventRegInfo* eventInfo
    ctypedef _n_PetscEventRegLog* PetscEventRegLog
    struct _PetscStageInfo:
        char *name
        PetscBool used
        PetscEventPerfLog eventLog
    ctypedef _PetscStageInfo PetscStageInfo
    struct _n_PetscStageLog:
        int numStages
        PetscStageInfo* stageInfo
        PetscEventRegLog eventLog

    ctypedef _n_PetscStageLog* PetscStageLog

    int PetscLogGetStageLog(PetscStageLog*)
    ctypedef int PetscLogStage
    int PetscLogStageRegister(char[],PetscLogStage*)
    int PetscLogStagePush(PetscLogStage)
    int PetscLogStagePop()
    int PetscLogStageSetActive(PetscLogStage,PetscBool)
    int PetscLogStageGetActive(PetscLogStage,PetscBool*)
    int PetscLogStageSetVisible(PetscLogStage,PetscBool)
    int PetscLogStageGetVisible(PetscLogStage,PetscBool*)
    int PetscLogStageGetId(char[],PetscLogStage*)

    ctypedef int PetscLogClass "PetscClassId"
    int PetscLogClassRegister"PetscClassIdRegister"(char[],PetscLogClass*)
    int PetscLogClassActivate"PetscLogEventActivateClass"(PetscLogClass)
    int PetscLogClassDeactivate"PetscLogEventDeactivateClass"(PetscLogClass)

    ctypedef int PetscLogEvent
    int PetscLogEventRegister(char[],PetscLogClass,PetscLogEvent*)
    int PetscLogEventBegin(PetscLogEvent,PetscObject,PetscObject,PetscObject,PetscObject)
    int PetscLogEventEnd(PetscLogEvent,PetscObject,PetscObject,PetscObject,PetscObject)

    int PetscLogEventActivate(PetscLogEvent)
    int PetscLogEventDeactivate(PetscLogEvent)
    int PetscLogEventSetActiveAll(PetscLogEvent,PetscBool)
    int PetscLogEventGetPerfInfo(PetscLogStage,PetscLogEvent,PetscEventPerfInfo*)

cdef extern from "custom.h" nogil:
    int PetscLogStageFindId(char[],PetscLogStage*)
    int PetscLogClassFindId(char[],PetscLogClass*)
    int PetscLogEventFindId(char[],PetscLogEvent*)
    int PetscLogStageFindName(PetscLogStage,char*[])
    int PetscLogClassFindName(PetscLogClass,char*[])
    int PetscLogEventFindName(PetscLogEvent,char*[])


cdef inline int event_args2objs(object args, PetscObject o[4]) except -1:
        o[0] = o[1] = o[2] = o[3] = NULL
        cdef Py_ssize_t i=0, n = len(args)
        cdef Object tmp = None
        if n > 4: n = 4
        for 0 <= i < n:
            tmp = args[i]
            if tmp is not None:
                o[i] = tmp.obj[0]
        return 0
