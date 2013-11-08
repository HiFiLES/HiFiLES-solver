#if defined EXTERN
#undef EXTERN
#endif
#if defined DATASET0MODULE
#define EXTERN
#else
#define EXTERN extern
#endif

/*
******************************************************************
******************************************************************
*******                                                   ********
******  (C) 1988-2008 Tecplot, Inc.                        *******
*******                                                   ********
******************************************************************
******************************************************************
*/

EXTERN void OutOfMemoryMsg(void);

/*
 * Turn on DEBUG_FIELDVALUES by default in any build with assertions on
 * (including checked builds), but allow turning this off with
 * NO_DEBUG_FIELDVALUES
 */
#if !defined NO_ASSERTS && !defined NO_DEBUG_FIELDVALUES && !defined DEBUG_FIELDVALUES
  #define DEBUG_FIELDVALUES
#endif

/* * DATASET FUNCTIONS * */

/* * FIELD DATA FUNCTIONS * */

/* FieldData_a is intentionally not defined to further
 * deter usage of this private structure */
struct _FieldData_a
{
  /*
   * IMPORTANT NOTE (CAM/RMS):
   *   The following two members may NEVER move as the variable ClientData
   *   embedded as the first member of the LoadOnDemandInfo_s structure is
   *   accessed by add-ons performing custom load-on-demand actions. Add-ons
   *   access the variable ClientData member via a macro that always expects
   *   it to be at the second pointer position of the field data pointer. The
   *   first pointer position is reserved for Tecplot's own Data pointer for
   *   fastest access.  See TecUtilDataValueCustomLOD() for details.
   */
  void               *Data;    /* !!! NEVER MOVE THIS !!! */
# if defined TECPLOTKERNEL
/* CORE SOURCE CODE REMOVED */
# else
  void *GetValueCallback[1]; /* ...this field is for TecIO only */
  void *SetValueCallback[1]; /* ...this field is for TecIO only */
# endif

  /* PRIVATE */
  FieldDataType_e     Type;
  ValueLocation_e     ValueLocation;
  LgIndex_t           RefCount;
  LgIndex_t           VarShareRefCount;
  LgIndex_t           NumValues;
# if defined TECPLOTKERNEL /* TecIO doesn't require these features yet. */
/* CORE SOURCE CODE REMOVED */
# endif
};


/* *
 * * NOTE: "FieldData_pa" here is an "abstract type".
 * * Any routines dealing with the internals workings
 * * of FieldData_pa must be in the same file as these
 * * routines
 * */

#if defined  USE_MACROS_FOR_FUNCTIONS
  #define USE_MACROS_FOR_FIELD_DATA_FUNCTIONS
#endif

/*
 * These are low-level (private) FD manipulation functions.  In
 * most cases, you should use some higher-level function.  These
 * macros are supplied for the dataset functions to use.
 */
#if defined USE_MACROS_FOR_FIELD_DATA_FUNCTIONS
  #define GetFieldDataType               GetFieldDataType_MACRO
  #define GetFieldDataGetFunction        GetFieldDataGetFunction_MACRO
  #define GetFieldDataSetFunction        GetFieldDataSetFunction_MACRO
  #define GetFieldDataNumValues          GetFieldDataNumValues_MACRO
  #define GetFieldDataValueLocation      GetFieldDataValueLocation_MACRO
  #define IsFieldDataDirectAccessAllowed IsFieldDataDirectAccessAllowed_MACRO
#else
  #define GetFieldDataType               GetFieldDataType_FUNC
  #define GetFieldDataGetFunction        GetFieldDataGetFunction_FUNC
  #define GetFieldDataSetFunction        GetFieldDataSetFunction_FUNC
  #define GetFieldDataNumValues          GetFieldDataNumValues_FUNC
  #define GetFieldDataValueLocation      GetFieldDataValueLocation_FUNC
  #define IsFieldDataDirectAccessAllowed IsFieldDataDirectAccessAllowed_FUNC
#endif

#define GetFieldDataType_MACRO(FieldData)          ((FieldData)->Type)
#if defined TECPLOTKERNEL
/* CORE SOURCE CODE REMOVED */
#else /* ...for TecIO only */
  #define GetFieldDataGetFunction_MACRO(FieldData)   ((FieldValueGetFunction_pf)(FieldData)->GetValueCallback[0])
  #define GetFieldDataSetFunction_MACRO(FieldData)   ((FieldValueSetFunction_pf)(FieldData)->SetValueCallback[0])
#endif
#define GetFieldDataNumValues_MACRO(FieldData)     ((FieldData)->NumValues)
#define GetFieldDataValueLocation_MACRO(FieldData) ((FieldData)->ValueLocation)

EXTERN double STDCALL GetFieldValueForFloat(const FieldData_pa fd, LgIndex_t pt);
EXTERN double STDCALL GetFieldValueForDouble(const FieldData_pa fd, LgIndex_t pt);
EXTERN double STDCALL GetFieldValueForInt32(const FieldData_pa fd, LgIndex_t pt);
EXTERN double STDCALL GetFieldValueForInt16(const FieldData_pa fd, LgIndex_t pt);
EXTERN double STDCALL GetFieldValueForByte(const FieldData_pa fd, LgIndex_t pt);
EXTERN double STDCALL GetFieldValueForBit(const FieldData_pa fd, LgIndex_t pt);

#if defined TECPLOTKERNEL
/* CORE SOURCE CODE REMOVED */
#else
  #define IsFieldDataDirectAccessAllowed_MACRO(FieldData) ((FieldData)->Data != NULL)
#endif

#if !defined USE_MACROS_FOR_FIELD_DATA_FUNCTIONS
  EXTERN FieldDataType_e GetFieldDataType_FUNC(FieldData_pa FieldData);
  EXTERN FieldValueGetFunction_pf GetFieldDataGetFunction_FUNC(FieldData_pa FieldData);
  EXTERN FieldValueSetFunction_pf GetFieldDataSetFunction_FUNC(FieldData_pa FieldData);
  EXTERN LgIndex_t GetFieldDataNumValues_FUNC(FieldData_pa FieldData);
  EXTERN ValueLocation_e GetFieldDataValueLocation_FUNC(FieldData_pa FieldData);
  EXTERN Boolean_t IsFieldDataDirectAccessAllowed_FUNC(FieldData_pa FieldData);
#endif


/*
 * Use separate types for reversed byte data than unreversed data so we
 * have better compiler checking.
 */
typedef UInt32_t FloatRev_t;
typedef UInt64_t DoubleRev_t;
typedef UInt16_t Int16Rev_t;
typedef UInt32_t Int32Rev_t;
typedef UInt64_t Int64Rev_t;


/*
 * Note: there are so many GetFieldData*Ptr functions because we
 * want a bunch of error checking.  The Type and TypeRev check
 * for that type.  The Byte, 2Byte, etc. just make sure it is
 * that type.
 * GetFieldDataVoidPtr checks nothing, and thus should only be
 * used with extreme caution (that is, checking the alignment
 * and byte order by hand).
 */
#if defined USE_MACROS_FOR_FIELD_DATA_FUNCTIONS
  #define GetFieldDataFloatPtr      GetFieldDataFloatPtr_MACRO
  #define GetFieldDataFloatRevPtr   GetFieldDataFloatRevPtr_MACRO
  #define GetFieldDataDoublePtr     GetFieldDataDoublePtr_MACRO
  #define GetFieldDataDoubleRevPtr  GetFieldDataDoubleRevPtr_MACRO
  #define GetFieldDataInt64Ptr      GetFieldDataInt64Ptr_MACRO
  #define GetFieldDataInt64RevPtr   GetFieldDataInt64RevPtr_MACRO
  #define GetFieldDataInt32Ptr      GetFieldDataInt32Ptr_MACRO
  #define GetFieldDataInt32RevPtr   GetFieldDataInt32RevPtr_MACRO
  #define GetFieldDataInt16Ptr      GetFieldDataInt16Ptr_MACRO
  #define GetFieldDataInt16RevPtr   GetFieldDataInt16RevPtr_MACRO
  #define GetFieldDataBytePtr       GetFieldDataBytePtr_MACRO
  #define GetFieldData2BytePtr      GetFieldData2BytePtr_MACRO
  #define GetFieldData4BytePtr      GetFieldData4BytePtr_MACRO
  #define GetFieldData8BytePtr      GetFieldData8BytePtr_MACRO
  #define GetFieldDataVoidPtr       GetFieldDataVoidPtr_MACRO /*danger:see above*/
#else
  #define GetFieldDataFloatPtr      GetFieldDataFloatPtr_FUNC
  #define GetFieldDataFloatRevPtr   GetFieldDataFloatRevPtr_FUNC
  #define GetFieldDataDoublePtr     GetFieldDataDoublePtr_FUNC
  #define GetFieldDataDoubleRevPtr  GetFieldDataDoubleRevPtr_FUNC
  #define GetFieldDataInt64Ptr      GetFieldDataInt64Ptr_FUNC
  #define GetFieldDataInt64RevPtr   GetFieldDataInt64RevPtr_FUNC
  #define GetFieldDataInt32Ptr      GetFieldDataInt32Ptr_FUNC
  #define GetFieldDataInt32RevPtr   GetFieldDataInt32RevPtr_FUNC
  #define GetFieldDataInt16Ptr      GetFieldDataInt16Ptr_FUNC
  #define GetFieldDataInt16RevPtr   GetFieldDataInt16RevPtr_FUNC
  #define GetFieldDataBytePtr       GetFieldDataBytePtr_FUNC
  #define GetFieldData2BytePtr      GetFieldData2BytePtr_FUNC
  #define GetFieldData4BytePtr      GetFieldData4BytePtr_FUNC
  #define GetFieldData8BytePtr      GetFieldData8BytePtr_FUNC
  #define GetFieldDataVoidPtr       GetFieldDataVoidPtr_FUNC /*danger:see above*/
#endif

#define GetFieldDataFloatPtr_MACRO(FieldData)     ((float *)((FieldData)->Data))
#define GetFieldDataFloatRevPtr_MACRO(FieldData)  ((FloatRev_t *)((FieldData)->Data))
#define GetFieldDataDoublePtr_MACRO(FieldData)    ((double *)((FieldData)->Data))
#define GetFieldDataDoubleRevPtr_MACRO(FieldData) ((DoubleRev_t *)((FieldData)->Data))
#define GetFieldDataInt64Ptr_MACRO(FieldData)     ((Int64_t *)((FieldData)->Data))
#define GetFieldDataInt64RevPtr_MACRO(FieldData)  ((Int64Rev_t *)((FieldData)->Data))
#define GetFieldDataInt32Ptr_MACRO(FieldData)     ((Int32_t *)((FieldData)->Data))
#define GetFieldDataInt32RevPtr_MACRO(FieldData)  ((Int32Rev_t *)((FieldData)->Data))
#define GetFieldDataInt16Ptr_MACRO(FieldData)     ((Int16_t *)((FieldData)->Data))
#define GetFieldDataInt16RevPtr_MACRO(FieldData)  ((Int16Rev_t *)((FieldData)->Data))
#define GetFieldDataBytePtr_MACRO(FieldData)      ((Byte_t *)((FieldData)->Data))
#define GetFieldData2BytePtr_MACRO(FieldData)     ((UInt16_t *)((FieldData)->Data))
#define GetFieldData4BytePtr_MACRO(FieldData)     ((UInt32_t *)((FieldData)->Data))
#define GetFieldData8BytePtr_MACRO(FieldData)     ((UInt64_t *)((FieldData)->Data))
#define GetFieldDataVoidPtr_MACRO(FieldData)      ((void *)((FieldData)->Data)) /*danger:see above*/

#if !defined USE_MACROS_FOR_FIELD_DATA_FUNCTIONS
  EXTERN float *GetFieldDataFloatPtr_FUNC(FieldData_pa fd);
  EXTERN FloatRev_t *GetFieldDataFloatRevPtr_FUNC(FieldData_pa fd);
  EXTERN double *GetFieldDataDoublePtr_FUNC(FieldData_pa fd);
  EXTERN DoubleRev_t *GetFieldDataDoubleRevPtr_FUNC(FieldData_pa fd);
  EXTERN Int64_t *GetFieldDataInt64Ptr_FUNC(FieldData_pa fd);
  EXTERN Int64Rev_t *GetFieldDataInt64RevPtr_FUNC(FieldData_pa fd);
  EXTERN Int32_t *GetFieldDataInt32Ptr_FUNC(FieldData_pa fd);
  EXTERN Int32Rev_t *GetFieldDataInt32RevPtr_FUNC(FieldData_pa fd);
  EXTERN Int16_t *GetFieldDataInt16Ptr_FUNC(FieldData_pa fd);
  EXTERN Int16Rev_t *GetFieldDataInt16RevPtr_FUNC(FieldData_pa fd);
  EXTERN Byte_t *GetFieldDataBytePtr_FUNC(FieldData_pa fd);
  EXTERN UInt16_t *GetFieldData2BytePtr_FUNC(FieldData_pa fd);
  EXTERN UInt32_t *GetFieldData4BytePtr_FUNC(FieldData_pa fd);
  EXTERN UInt64_t *GetFieldData8BytePtr_FUNC(FieldData_pa fd);
  EXTERN void *GetFieldDataVoidPtr_FUNC(FieldData_pa fd); /*danger:see above*/
#endif

/**
 */
EXTERN FieldData_pa AllocScratchNodalFieldDataPtr(LgIndex_t       NumValues,
                                                  FieldDataType_e Type,
                                                  Boolean_t       ShowErrMsg);

/**
 */
EXTERN void DeallocScratchNodalFieldDataPtr(FieldData_pa *ScratchFieldData);

/**
 * Assume that indexrange has already been converted to the actual indices.
 */
EXTERN void CalcFieldDataMinMaxUsingRange(FieldData_pa  field_data,
                                          double       *min_ptr,
                                          double       *max_ptr,
                                          LgIndex_t     startindex,
                                          IndexRange_s *indexrange);

#if defined TECPLOTKERNEL
/* CORE SOURCE CODE REMOVED */
#endif

#if defined TECPLOTKERNEL
/* CORE SOURCE CODE REMOVED */
#endif

#if defined TECPLOTKERNEL
/* CORE SOURCE CODE REMOVED */
#endif

#if defined TECPLOTKERNEL
/* CORE SOURCE CODE REMOVED */
#endif

/**
 */
EXTERN void CopyTypedValueArray(FieldDataType_e  ValueType,
                                void            *DstArray,
                                LgIndex_t        DstStart,
                                LgIndex_t        DstSkip,
                                void            *SrcArray,
                                LgIndex_t        SrcStart,
                                LgIndex_t        SrcEnd,
                                LgIndex_t        SrcSkip);

EXTERN void CopyUnalignedTypedValueArray(FieldDataType_e  ValueType,
                                         void            *DstArray,
                                         LgIndex_t        DstStart,
                                         LgIndex_t        DstSkip,
                                         void            *SrcArray,
                                         LgIndex_t        SrcStart,
                                         LgIndex_t        SrcEnd,
                                         LgIndex_t        SrcSkip);

EXTERN void SwapBytesInTypedValueArray(FieldDataType_e  ValueType,
                                       void            *SrcArray,
                                       LgIndex_t        SrcStart,
                                       LgIndex_t        SrcEnd,
                                       LgIndex_t        SrcSkip);

EXTERN void SwapBytesInUnalignedTypedValueArray(FieldDataType_e  ValueType,
                                                void            *SrcArray,
                                                LgIndex_t        SrcStart,
                                                LgIndex_t        SrcEnd,
                                                LgIndex_t        SrcSkip);


/*
 * Copies values from "src" to "dst".  "src" or "dst" may
 * be differing types.  Either or both may be V3D data pointers.
 */
EXTERN void CopyFieldDataRange(FieldData_pa dst,
                               LgIndex_t    dst_start,
                               LgIndex_t    dst_skip,
                               FieldData_pa src,
                               LgIndex_t    src_start,
                               LgIndex_t    src_end, /* -1 means last point */
                               LgIndex_t    src_skip);

/*
 * Copy all values in field data
 */
EXTERN void CopyFieldData(FieldData_pa dst,
                          FieldData_pa src);

/*
 * Like CopyFieldData except for single value.
 */
EXTERN void CopyFieldValue(FieldData_pa dst,
                           LgIndex_t    dstindex,
                           FieldData_pa src,
                           LgIndex_t    srcindex);


#if defined TECPLOTKERNEL
/* CORE SOURCE CODE REMOVED */
#endif

#if defined TECPLOTKERNEL
/* CORE SOURCE CODE REMOVED */
#endif

#if defined TECPLOTKERNEL
/* CORE SOURCE CODE REMOVED */
#endif

#if defined TECPLOTKERNEL
/* CORE SOURCE CODE REMOVED */
#endif

/*
 * Sets all values in the field data pointer "field_data"
 * to zero.
 */
EXTERN void SetFieldDataPtrToAllZeros(FieldData_pa field_data);


/* * NODE MAP FUNCTIONS * */

/* *
 * * NOTE: "NodeMap_pa" here is an "abstract type".
 * * Any routines dealing with the internals workings
 * * of NodeMap_pa must be in the same file as these
 * * routines.
 * */

/* NodeMap_a is intentionally not defined to further
 * deter usage of this private structure */
struct _NodeMap_a
  {
    /* * PRIVATE * */
    LgIndex_t   RefCount;
    LgIndex_t   NumElements;
    ZoneType_e  ElementType;
    EntIndex_t  PtsPerElement;
    Boolean_t   HasOrphanedNodes;
    Boolean_t   IsValidHasOrphanedNodes; /* guards HasOrphanedNodes flag */
    NodeMap_t  *Data;
  };

/*
 * These are low-level (private) NM manipulation functions.  In
 * most cases, you should use some higher-level function.  These
 * macros are supplied for the dataset functions to use.  In
 * particular, these macros have no error checking.
 */
#define GetNMRefCount(nm)      ((nm)->RefCount)
#define GetNMElementType(nm)   ((nm)->ElementType)
#define GetNMPtsPerElement(nm) ((nm)->PtsPerElement)
#define GetNMNumElements(nm)   ((nm)->NumElements)
#define GetNMPtr(nm)           ((NodeMap_t *)(nm)->Data)

#define NodeMapHasOrphanedNodes(NodeMap) ((NodeMap)->HasOrphanedNodes)

  /*
 * Allocates an FE node map pointer for "num_elements"
 * of element type "element_type" and node map type
 * "node_map_type".  Returns NULL if out of memory.
 */
EXTERN NodeMap_pa AllocNodeMapPtr(LgIndex_t     num_elements,
                                  ZoneType_e    element_type,
                                  Boolean_t     show_error_msg);

/*
 * Frees all memory associated with FE node map pointer
 * "*node_map" and sets "*node_map" to NULL.
 */
EXTERN void DeallocNodeMapPtr(NodeMap_pa *node_map);

/*
 * Copies node map from "src" to "dst".  "src" or "dst" may
 * be differing node map ptr types.
 */
EXTERN void CopyNodeMap(NodeMap_pa dst,
                        NodeMap_pa src);



/*
 * GetFieldValue macro
 */
#if !defined GET_FIELD_VALUE_BY_VIRTUAL_FUNCTION && \
    !defined GET_FIELD_VALUE_BY_FLOAT_ONLY_MACRO && \
    !defined GET_FIELD_VALUE_BY_DOUBLE_ONLY_MACRO && \
    !defined GET_FIELD_VALUE_BY_FLOAT_AND_DOUBLE_MACRO
  #if !defined NO_ASSERTS || defined DEBUG_FIELDVALUES
    #define GET_FIELD_VALUE_BY_VIRTUAL_FUNCTION
  #else
    #define GET_FIELD_VALUE_BY_FLOAT_AND_DOUBLE_MACRO
  #endif
#endif

#if defined GET_FIELD_VALUE_BY_VIRTUAL_FUNCTION
  #define GetFieldValue(fd,pt) ((GetFieldDataGetFunction(fd))((fd),(pt)))
#elif defined GET_FIELD_VALUE_BY_FLOAT_ONLY_MACRO
  #define GetFieldValue(fd,pt) (GetFieldDataGetFunction(fd)==GetFieldValueForFloat \
                                ?GetFieldDataFloatPtr(fd)[(pt)] \
                                :(GetFieldDataGetFunction(fd))((fd),(pt)))
#elif defined GET_FIELD_VALUE_BY_DOUBLE_ONLY_MACRO
  #define GetFieldValue(fd,pt) (GetFieldDataGetFunction(fd)==GetFieldValueForDouble \
                                ?GetFieldDataDoublePtr(fd)[(pt)] \
                                :(GetFieldDataGetFunction(fd))((fd),(pt)))
#elif defined GET_FIELD_VALUE_BY_FLOAT_AND_DOUBLE_MACRO
  #define GetFieldValue(fd,pt) (GetFieldDataGetFunction(fd)==GetFieldValueForFloat \
                                ?GetFieldDataFloatPtr(fd)[(pt)] \
                                :GetFieldDataGetFunction(fd)==GetFieldValueForDouble \
                                ?GetFieldDataDoublePtr(fd)[(pt)] \
                                :(GetFieldDataGetFunction(fd))((fd),(pt)))
#else
  #error "Need to define one of FIELD_VALUE_MACRO constants"
#endif


/*
 * SetFieldValue macro
 */
#define SetFieldValue(fd,pt,val) ((GetFieldDataSetFunction(fd))((fd),(pt),(val)))


#if defined TECPLOTKERNEL
/* CORE SOURCE CODE REMOVED */
#endif /* TECPLOTKERNEL */

#if defined _DEBUG
  #define USEFUNCTIONSFORNODEVALUES
#endif

#if defined USEFUNCTIONSFORNODEVALUES
/*
 * Gets the node for corner "corner" in element "element"
 * in FE node map "node_map"
 */
EXTERN NodeMap_t GetNodeValue(NodeMap_pa  node_map,
                              LgIndex_t   element,
                              LgIndex_t   corner);

/*
 * Gets the node at the offset "offset" in FE node map "node_map"
 */
EXTERN NodeMap_t GetNodeValueAtOffset(NodeMap_pa  node_map,
                                      LgIndex_t   offset);

/*
 * Sets "node" as the node for corner "corner" in element
 * "element" in FE node map "node_map"
 */
EXTERN void SetNodeValue(NodeMap_pa node_map,
                         LgIndex_t  element,
                         LgIndex_t  corner,
                         NodeMap_t  node);

/*
 * Gets the node at the offset "offset" in FE node map "node_map"
 */
EXTERN void SetNodeValueAtOffset(NodeMap_pa  node_map,
                                 LgIndex_t   offset,
                                 NodeMap_t   node);
#else
  #if defined __cplusplus
     inline NodeMap_t GetNodeValue(NodeMap_pa  node_map,
                                   LgIndex_t   element,
                                   LgIndex_t   corner)
     {
       LgIndex_t pos = element*GetNMPtsPerElement(node_map) + corner;
       return GetNMPtr(node_map)[pos];
     } /* GetNodeValue() */

     inline void SetNodeValue(NodeMap_pa  node_map,
                              LgIndex_t   element,
                              LgIndex_t   corner,
                              NodeMap_t   node)
     {
       LgIndex_t pos = element*GetNMPtsPerElement(node_map) + corner;
       GetNMPtr(node_map)[pos] = node;
     } /* SetNodeValue() */

     inline NodeMap_t GetNodeValueAtOffset(NodeMap_pa  node_map,
                                           LgIndex_t   offset)
     {
       return GetNMPtr(node_map)[offset];
     } /* GetNodeValueAtOffset() */

     inline void SetNodeValueAtOffset(NodeMap_pa  node_map,
                                      LgIndex_t   offset,
                                      NodeMap_t   node)
     {
       GetNMPtr(node_map)[offset] = node;
     } /* SetNodeValueAtOffset() */
  #else
    #define GetNodeValue(nm,element,corner) (((NodeMap_t *)(nm)->Data)[(nm)->PtsPerElement*(element)+(corner)])
    #define SetNodeValue(nm,element,corner,node) (((NodeMap_t *)(nm)->Data)[(nm)->PtsPerElement*(element)+(corner)] = (node))
    #define GetNodeValueAtOffset(nm,offset) (((NodeMap_t *)(nm)->Data)[(offset)])
    #define SetNodeValueAtOffset(nm,offset,node) (((NodeMap_t *)(nm)->Data)[(offset)] = (node))
  #endif
#endif

#if defined TECPLOTKERNEL
/* CORE SOURCE CODE REMOVED */
#if !defined NO_ASSERTS
#endif
#endif /* TECPLOTKERNEL */

#if defined TECPLOTKERNEL
/* CORE SOURCE CODE REMOVED */
#endif /* TECPLOTKERNEL */

#if defined TECPLOTKERNEL
/* CORE SOURCE CODE REMOVED */
#endif
