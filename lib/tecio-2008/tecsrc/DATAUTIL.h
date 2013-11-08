/*
 * DATAUTIL.h : COPYRIGHT (C)1987-2002 Tecplot, Inc.          
 *                 ALL RIGHTS RESERVED
 *
 * NOTE:  THIS MODULE NOW IS PART OF THE TECPLOT SOURCE
 *        ONLY EDIT THIS IN THE MAIN TECPLOT SOURCE DIRECTORY.
 *
 *
 */
#ifndef DATAUTIL_H
#define DATAUTIL_H
#define DATAUTIL_VERSION 61

#if defined MAKEARCHIVE
extern void InitInputSpecs(void);
#endif


/*
 *
 * Read a binary tecplot datafile.  
 *
 * If GetHeaderInfoOnly is TRUE then only the header info
 * is retrieved.
 *
 * Variable                Description
 * ---------------------------------------------------------------
 * GetHeaderInfoOnly       Return only the header info from the datafile.
 * FName                   Name of the file to read.
 * IVersion                Returns version of the input file.
 * DataSetTitle            Allocates space for and returns dataset title.
 * NumZones                Returns the number of zones.
 * NumVars                 Returns the number of variables.
 * VarNames                Allocates space for and returns the var names.
 * ZoneInfo                Allocates space for and returns the zone information.
 * NumUserRec              Returns the number of user records in the binary datafile.
 * MaxCharsUserRec         Maximum number of characters allowed in user recs.
 * UserRec                 Allocates space for and returns the user records.
 * RawDataspaceAllocated   TRUE = calling program has alloced space for the raw data.
 *                         FALSE= let ReadTec allocate space for the raw data.
 *                         (Only used if GetHeaderInfoOnly is FALSE)
 * NodeMap                 Finite Element connectivity information.  ReadTec
 *                         will allocate the space for you if RawDataspaceAllocated is
 *                         FALSE.
 * VDataBase               Raw field data loaded into double arrays.  ReadTec
 *                         will allocate the space for you if RawDataspaceAllocated is
 *                         FALSE.  If RawDataspaceAllocated is TRUE then ReadTec will
 *                         only load the arrays that have non NULL addresses.
 *
 */


LIBFUNCTION Boolean_t STDCALL ReadTec(Boolean_t       GetHeaderInfoOnly,
                                      char           *FName,
                                      short          *IVersion,
                                      char          **DataSetTitle,
                                      EntIndex_t     *NumZones,
                                      EntIndex_t     *NumVars,
                                      StringList_pa  *VarNames,
                                      StringList_pa  *ZoneNames,
                                      LgIndex_t     **NumPtsI,
                                      LgIndex_t     **NumPtsJ,
                                      LgIndex_t     **NumPtsK,
                                      ZoneType_e    **ZoneType,
                                      StringList_pa  *UserRec,
                                      Boolean_t       RawDataspaceAllocated,
                                      NodeMap_t    ***NodeMap, 
                                      double       ***VDataBase);

LIBFUNCTION void * STDCALL TecAlloc(size_t size);

LIBFUNCTION void STDCALL TecFree(void *ptr);


#endif /* !DATAUTIL_H */
