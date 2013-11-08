#if defined EXTERN
#undef EXTERN
#endif
#if defined DATASHRMODULE
#define EXTERN
#else
#define EXTERN extern
#endif

/* 
*****************************************************************
*****************************************************************
*******                                                  ********
****** Copyright (C) 1988-2008 Tecplot, Inc.              *******
*******                                                  ********
*****************************************************************
*****************************************************************
*/

#if defined TECPLOTKERNEL
/* CORE SOURCE CODE REMOVED */
#endif /* TECPLOTKERNEL */

/*
 * General set of macros for reference count mananagement.
 */
#define IncStructureReference(V)  ((V)->RefCount++)
#define DecStructureReference(V)  ((V)->RefCount--)
#define IsStructureShared(V)      ((V)->RefCount > 1)
#define IsStructureReferenced(V)  ((V)->RefCount > 0)

/*
 * Special set of macros for field data that is having variable sharing between
 * zones tracked. Field data maintains two reference counts: The first,
 * RefCount, is used to keep track of when the field data needs to be
 * deallocated; the second, VarShareRefCount, is used to track variable sharing
 * between zones.
 */
#define IncVarStructureReference(V)  ((V)->VarShareRefCount++)
#define DecVarStructureReference(V)  ((V)->VarShareRefCount--)
#define IsVarStructureShared(V)      ((V)->VarShareRefCount > 1)
#define IsVarStructureReferenced(V)  ((V)->VarShareRefCount > 0)


#if defined TECPLOTKERNEL
/* CORE SOURCE CODE REMOVED */
#endif /* TECPLOTKERNEL */
