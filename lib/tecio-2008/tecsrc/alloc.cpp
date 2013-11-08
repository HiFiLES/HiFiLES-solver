#include "stdafx.h"
#include "MASTER.h"

#define TECPLOTENGINEMODULE

/*
******************************************************************
******************************************************************
*******                                                   ********
******  (C) 1988-2008 Tecplot, Inc.                        *******
*******                                                   ********
******************************************************************
******************************************************************
*/

#define ALLOCMODULE
#include "GLOBAL.h"
#include "ALLOC.h"
#include "TASSERT.h"
#if defined TECPLOTKERNEL
/* CORE SOURCE CODE REMOVED */
#endif

#if defined UNIXX
static size_t MaxSizeAllocated     = 0;
static size_t NumMallocInvocations = 0;
static size_t NumFreeInvocations   = 0;
static size_t MaxActiveAllocs      = 0;
static size_t NumActiveAllocs      = 0;
#  if !defined DEBUG_ALLOC
static size_t TotalBytesAllocated  = 0;
#  endif
#endif

#include <stdlib.h>

#define ENDADDR(dataaddr,size) ((char *)(dataaddr)+size)
#define ENDCHAR   ((char)0xFC)
#define ALLOCAMOUNT(size) ((size)+12)

#if defined TECPLOTKERNEL
/* CORE SOURCE CODE REMOVED */
#endif

#ifdef DEBUG_ALLOC

#if defined TECPLOTKERNEL
/* CORE SOURCE CODE REMOVED */
#endif

static size_t    NumBytesAllocated   = 0;
static size_t    MaxBytesAllocated   = 0;
static Boolean_t DebugAllocInitialized = FALSE;
static FILE     *DebugAllocLogFile = NULL;

#if defined CRUDEALLOCVERIFY
# if !defined MAXACTIVEALLOCS
#   define MAXACTIVEALLOCS 20000
# endif
#define MAX_ALLOC_DESC_LEN 20

typedef void *AllocPtr;
typedef char  AllocDesc[MAX_ALLOC_DESC_LEN+1];

static Boolean_t UseActiveAllocLog = TRUE;
static AllocPtr  ActiveAllocs[MAXACTIVEALLOCS];
static size_t    ActiveAllocSizes[MAXACTIVEALLOCS];
static AllocDesc ActiveAllocFile[MAXACTIVEALLOCS];
static AllocDesc ActiveAllocDesc[MAXACTIVEALLOCS];




#define STARTSIZEADDR(addr) (void *)(((char *)(addr)-8))

/**
 */
static void AllocErrMsg(const char *EMsg)
{
  PendingAllocErrMsg = TRUE;

  /* we don't use ErrMsg because it calls ALLOC causing infinite recursion */
  fprintf(stderr, "%s\n", EMsg);
  PrintCurBacktrace(stderr, MAX_BACKTRACE_DEPTH);
}

/**
 */
size_t GetArraySize(void *Adr)
{
  void *adjptr = STARTSIZEADDR(Adr);
  size_t Size = *(size_t *)adjptr;
  return (Size);
}

/**
 */
static void LockedCheckActiveAllocs(void)
{
  if (UseActiveAllocLog)
    {
      for (size_t ii = 0; ii < NumActiveAllocs; ii++)
        {
          void   *adjptr    = STARTSIZEADDR(ActiveAllocs[ii]);
          char   *endptr    = ENDADDR(ActiveAllocs[ii],ActiveAllocSizes[ii]);
          size_t  startsize = *(size_t *)adjptr;
          if (startsize != ActiveAllocSizes[ii])
            {
              char EMsg[100*MAX_SIZEOFUTF8CHAR];
              sprintf(EMsg,"Internal Error: Memory size tag has been corrupted: %s,%s",
                      ActiveAllocFile[ii],
                      ActiveAllocDesc[ii]);
              AllocErrMsg(EMsg);
              return;
            }
          else if ((endptr[0] != ENDCHAR) ||
                   (endptr[1] != ENDCHAR) ||
                   (endptr[2] != ENDCHAR) ||
                   (endptr[3] != ENDCHAR))
            {
              char EMsg[100*MAX_SIZEOFUTF8CHAR];
              sprintf(EMsg,"Internal Error: Memory end tag has been corrupted: %s,%s",
                      ActiveAllocFile[ii],
                      ActiveAllocDesc[ii]);
              AllocErrMsg(EMsg);
              return;
            }
        }
    }
}

/**
 */
void CheckActiveAllocs(void)
{
  if (UseActiveAllocLog)
    {
#     if defined TECPLOTKERNEL
/* CORE SOURCE CODE REMOVED */
#     endif

      LockedCheckActiveAllocs();

#     if defined TECPLOTKERNEL
/* CORE SOURCE CODE REMOVED */
#     endif
    }
}
#endif

/**
 */
static void InitDebugAlloc(void)
{
  REQUIRE(!DebugAllocInitialized && DebugAllocLogFile == NULL);

  char *DebugAllocLogFileName = GETENV("DEBUGALLOCLOGFILE");
  UseActiveAllocLog = (DebugAllocLogFileName != NULL &&
                       strcmp(DebugAllocLogFileName, "/dev/null") != 0);
  if (UseActiveAllocLog)
    {
      if (strcmp(DebugAllocLogFileName, "stdout") == 0)
        DebugAllocLogFile = stdout;
      else if (strcmp(DebugAllocLogFileName, "stderr") == 0)
        DebugAllocLogFile = stderr;
      else
        {
          DebugAllocLogFile = FOPEN(DebugAllocLogFileName, "w");
          if (DebugAllocLogFile == NULL)
            DebugAllocLogFile = stderr;
        }
    }
  else
    {
      DebugAllocLogFile = NULL;
    }
  
  DebugAllocInitialized = TRUE;

  ENSURE(DebugAllocInitialized);
  ENSURE((!UseActiveAllocLog && DebugAllocLogFile == NULL) ||
         (UseActiveAllocLog  && DebugAllocLogFile != NULL));
}

/**
 */
static void MopupDebugAllocLogFile(void)
{
  if (DebugAllocLogFile != NULL   &&
      DebugAllocLogFile != stdout &&
      DebugAllocLogFile != stderr)
    FCLOSE(DebugAllocLogFile);
}

/**
 */
void PrintAllocStats(void)
{
  if (!DebugAllocInitialized)
    InitDebugAlloc();

  if (UseActiveAllocLog)
    {
      fprintf(DebugAllocLogFile,"\n\n\n\n-------------- MEMORY ALLOCATION STATISTICS -------------\n\n");
      fprintf(DebugAllocLogFile,"    Maximum bytes allocated   = %ld\n",(long)MaxBytesAllocated);
      fprintf(DebugAllocLogFile,"    Maximum size  allocated   = %ld\n",(long)MaxSizeAllocated);
#if defined CRUDEALLOCVERIFY
      fprintf(DebugAllocLogFile,"    Maximum active allocs     = %ld\n",(long)MaxActiveAllocs);
#endif
      fprintf(DebugAllocLogFile,"    Number of calls to alloc  = %ld\n",(long)NumMallocInvocations);
      fprintf(DebugAllocLogFile,"    Number of bytes not freed = %ld\n\n",(long)NumBytesAllocated);
    }

  MopupDebugAllocLogFile();
}

/**
 */
void *newmalloc(size_t       size, 
                const char  *file, 
                long         line, 
                const char  *desc)
{
  REQUIRE(size > 0);
  REQUIRE(VALID_NON_ZERO_LEN_STR(file));
  REQUIRE(line >= 1);
  REQUIRE(VALID_NON_ZERO_LEN_STR(desc));

  #if defined TECPLOTKERNEL && defined CRUDEALLOCVERIFY
    if ( Thread_ThreadingIsInitialized() && UseActiveAllocLog )
      Thread_LockMutex(AllocStatsMutex);
  #endif

  if (!DebugAllocInitialized)
    InitDebugAlloc();

  double DAmount = ALLOCAMOUNT((double)size);
  void *ptr = NULL;
  if (DAmount <= MAXINDEX)
    ptr = (void *)malloc(ALLOCAMOUNT(size));

  if (ptr)
    {
      *(size_t *)ptr = size;
      ptr = (void *)(((char *)ptr)+8);
      char *endptr = ENDADDR(ptr,size);
      endptr[0] = ENDCHAR;
      endptr[1] = ENDCHAR;
      endptr[2] = ENDCHAR;
      endptr[3] = ENDCHAR;
      NumBytesAllocated += size;
      MaxBytesAllocated = MAX(MaxBytesAllocated,NumBytesAllocated);
      MaxSizeAllocated  = MAX(MaxSizeAllocated,size);
      NumMallocInvocations++;

      if (UseActiveAllocLog)
        {
          fprintf(DebugAllocLogFile, "%8.8lx %6ld MALLOC %6ld %6ld %14s:%-5ld %s\n",
                  (long)ptr, (long)size, (long)NumBytesAllocated,
                  (long)NumMallocInvocations, file, (long)line, desc);
          fflush(DebugAllocLogFile);
        }

#if defined CRUDEALLOCVERIFY
      if (UseActiveAllocLog)
        {
          LockedCheckActiveAllocs();
          if (NumActiveAllocs == MAXACTIVEALLOCS-1)
            {
              UseActiveAllocLog = FALSE;
              fprintf(DebugAllocLogFile, "\n\n\n");
              fprintf(DebugAllocLogFile, "*************************************************\n");
              fprintf(DebugAllocLogFile, "* Max Active Logs exceeded.  Tracking shut down *\n");
              fprintf(DebugAllocLogFile, "*************************************************\n");

              /* we don't call Warning() because it uses newmalloc */
              fprintf(stderr, "Alloc verifier maxed out.");
            }
          else
            {
              int MaxLeftOver = 0;

              ActiveAllocs[NumActiveAllocs]     = ptr;
              ActiveAllocSizes[NumActiveAllocs] = size;
              const char *FileName = NULL;
              if (strlen(file) > MAX_ALLOC_DESC_LEN)
                FileName = strrchr(file,'/');
              if (FileName != NULL &&
                  (MaxLeftOver = MAX_ALLOC_DESC_LEN - (strlen("...")+strlen(FileName))) >= 0)
                {
                  if (MaxLeftOver > 0)
                    {
                      strncpy(ActiveAllocFile[NumActiveAllocs], file, MaxLeftOver);
                      ActiveAllocFile[NumActiveAllocs][MaxLeftOver] = '\0';
                    }
                  else
                    {
                      ActiveAllocFile[NumActiveAllocs][0] = '\0';
                    }
                  strcat(ActiveAllocFile[NumActiveAllocs], "...");
                  strcat(ActiveAllocFile[NumActiveAllocs], FileName);
                }
              else
                {
                  strncpy(ActiveAllocFile[NumActiveAllocs],file,MAX_ALLOC_DESC_LEN);
                  ActiveAllocFile[NumActiveAllocs][MAX_ALLOC_DESC_LEN] = '\0';
                }
              strncpy(ActiveAllocDesc[NumActiveAllocs],desc,MAX_ALLOC_DESC_LEN);
              ActiveAllocDesc[NumActiveAllocs][MAX_ALLOC_DESC_LEN] = '\0';
              NumActiveAllocs++;
              MaxActiveAllocs = MAX(MaxActiveAllocs,NumActiveAllocs);
            }
       }
#endif
    }

  ENSURE(VALID_REF(ptr) || (ptr == NULL));

  #if defined TECPLOTKERNEL && defined CRUDEALLOCVERIFY
    if ( Thread_ThreadingIsInitialized() && UseActiveAllocLog )
      Thread_UnlockMutex(AllocStatsMutex);
  #endif

  return ptr;
} /* newmalloc() */

/**
 * Assign 0xdeadbeef to each word of memory and then
 * assign remaining bytes with 0xff.
 */
static void SetMemoryToDeadBeef(void   *Address,
                                size_t NumBytes)
{
  REQUIRE(VALID_REF(Address));
  REQUIRE(NumBytes >= 1);

  size_t NumBytesPer32BitWord = 4;
  size_t Num32BitWords = NumBytes / NumBytesPer32BitWord;
  size_t NumBytesRemaining = NumBytes % NumBytesPer32BitWord;

  for (size_t Word = 0; Word < Num32BitWords; Word++)
    {
      char *WordPtr = ((char *)Address) + Word*NumBytesPer32BitWord;
#if defined MACHINE_DOES_INTEL_ORDER
      WordPtr[0] = 0xef;
      WordPtr[1] = 0xbe;
      WordPtr[2] = 0xad;
      WordPtr[3] = 0xde;
#else
      WordPtr[0] = 0xde;
      WordPtr[1] = 0xad;
      WordPtr[2] = 0xbe;
      WordPtr[3] = 0xef;
#endif
    }

  CHECK(0 <= NumBytesRemaining && NumBytesRemaining <= 3);
  for (size_t Byte = 0; Byte < NumBytesRemaining; Byte++)
    {
      char *BytePtr = ((char *)Address) +
                       Num32BitWords * NumBytesPer32BitWord + Byte;
      *BytePtr = 0xff;
    }
}

/**
 */
void newfree(void       *ptr, 
             const char *file, 
             long        line, 
             const char *desc)
{
  REQUIRE(VALID_REF(ptr));
  REQUIRE(VALID_NON_ZERO_LEN_STR(file));
  REQUIRE(line >= 1);
  REQUIRE(VALID_NON_ZERO_LEN_STR(desc));

  #if defined TECPLOTKERNEL && defined CRUDEALLOCVERIFY
  if ( Thread_ThreadingIsInitialized() && UseActiveAllocLog )
    Thread_LockMutex(AllocStatsMutex);
  #endif

  if (!DebugAllocInitialized)
    InitDebugAlloc();

  void *adjptr = STARTSIZEADDR(ptr);
  size_t size = *(size_t *)adjptr;
#if defined CRUDEALLOCVERIFY
  if (UseActiveAllocLog)
    {
      LockedCheckActiveAllocs();

      /*
       * make the assumption that the item was recently
       * allocated and therefore near the end of the list
       */
      int ii = (int)NumActiveAllocs-1;
      while (ii >= 0 && ActiveAllocs[ii] != ptr)
        ii--;
      if (ii == -1)
        {
          AllocErrMsg("Attempt to free unallocated memory");
          return;
        }

      /* remove the item from the alloc log */
      NumActiveAllocs--;
      while (ii < (int)NumActiveAllocs)
        {
          ActiveAllocs[ii]     = ActiveAllocs[ii+1];
          ActiveAllocSizes[ii] = ActiveAllocSizes[ii+1];
          strcpy(ActiveAllocFile[ii],ActiveAllocFile[ii+1]);
          strcpy(ActiveAllocDesc[ii],ActiveAllocDesc[ii+1]);
          ii++;
        }
    }
#endif
  if (size <= 0)
    {
      AllocErrMsg("Internal Error, FREE with size <= 0");
      return;
    }
  else
    {
      NumBytesAllocated -= size;
      NumFreeInvocations++;
      if (UseActiveAllocLog)
        fprintf(DebugAllocLogFile, "%8.8lx %6ld FREE   %6ld %6ld %14s:%-5ld %s\n",
                (long)ptr, (long)size, (long)NumBytesAllocated,
                (long)NumFreeInvocations, file, (long)line, desc);
    }
  if (UseActiveAllocLog)
    fflush(DebugAllocLogFile);

  #if defined TECPLOTKERNEL && defined CRUDEALLOCVERIFY
  if ( Thread_ThreadingIsInitialized() && UseActiveAllocLog )
    Thread_UnlockMutex(AllocStatsMutex);
  #endif

  SetMemoryToDeadBeef(adjptr, ALLOCAMOUNT(size));
  free(adjptr);
} /* newfree() */

#else

#  if defined TECPLOTKERNEL
/* CORE SOURCE CODE REMOVED */
#    if defined UNIXX
#     if defined TECPLOTKERNEL
#     endif
#     if defined TECPLOTKERNEL
#     endif
#   else
#    endif
#  endif
#endif

#if defined TECPLOTKERNEL
/* CORE SOURCE CODE REMOVED */
#endif

#if defined TECPLOTKERNEL
/* CORE SOURCE CODE REMOVED */
#endif

#if defined MSWIN && defined ALLOC_HEAP

#define HEAPMIN 512
/**
 */
void *MSWinAlloc(DWORD nSize)
{
  long *pMem = NULL;
  if ( nSize < HEAPMIN )
    pMem = (long *)malloc(sizeof(long)+nSize);
  else
    pMem =(long *)HeapAlloc(GetProcessHeap(), NULL, sizeof(long)+nSize);
  if ( pMem )
    pMem[0] = nSize;
  return (void *)&(pMem[1]);
}

/**
 */
void MSWinFree(void *pMem)
{
  REQUIRE(VALID_REF(pMem));
  if ( pMem )
    {
      long *pMemLong = &(((long *)pMem)[-1]);
      if ( pMemLong[0] < HEAPMIN )
        free((void *)pMemLong);
      else
        HeapFree(GetProcessHeap(), NULL, (void *)pMemLong);
    }
}

#endif
