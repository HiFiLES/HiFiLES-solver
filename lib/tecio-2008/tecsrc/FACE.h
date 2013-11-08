/*
******************************************************************
******************************************************************
*******                                                   ********
******  (C) 1988-2008 Tecplot, Inc.                        *******
*******                                                   ********
******************************************************************
******************************************************************
*/
#ifndef _FACE_H_
#define _FACE_H_

#if defined EXTERN
#undef EXTERN
#endif
#if defined FACEMODULE
#define EXTERN
#else
#define EXTERN extern
#endif

namespace tecplot { namespace kernel {
    class SubElemValueProducerInterface;
}}

#define MAX_NODES_PER_FACE    4
#define MAX_NODES_PER_ELEMENT 8

/**
 */
#define CELL_FACE_IS_LOGICALLY_COLLAPSED(I1,I2,I3,I4) \
          (((I1) == (I2) && (I3) == (I4)) || \
           ((I1) == (I4) && (I2) == (I3)) || \
           ((I1) == (I3))                 || \
           ((I2) == (I4)))

/**
 * IMPORTANT NOTE:
 *   A face obscuration of FaceObscuration_LogicallyObscured means that the
 *   face is entirely obscured by either an implicit neighbor for inside faces
 *   of ordered data or an auto generated neighbor for finite element data. In
 *   either case, logical obscuration is not considered if user defined
 *   neighbors have been specified for the face. Therefore, interior faces of
 *   ordered data can have an indication of FaceObscuration_PartiallyObscured.
 */
typedef enum
  {
    FaceObscuration_NotObscured,
    FaceObscuration_PartiallyObscured,
    FaceObscuration_EntirelyObscured,
    FaceObscuration_LogicallyObscured,
    END_FaceObscuration_e,
    FaceObscuration_Invalid = BadEnumValue
  } FaceObscuration_e;

/**
 */
struct _AutoGenFaceNbr_a
  {
    LgIndex_t NumConnections;
    LgIndex_t NumBoundaryFaces;
    LgIndex_t NumCollapsedFaces;
    LgIndex_t *CellList; /*[TotalNumFaces]*/
  };

/**
 */
struct _UserDefFaceNbr_a
  {
    LgIndex_t  NumConnections;
    LgIndex_t  NumBoundaryFaces;
    Set_pa     NeighborsCompletelyObscure; /*[TotalNumFaces]*/
    LgIndex_t  *FaceToCellListMap;         /*[TotalNumFaces + 1]*/
    Set_pa     IsPerfectNeighbor;          /*[this->NumConnections]*/
    LgIndex_t  *CellList;                  /*[this->NumConnections]*/
    EntIndex_t *ZoneList;                  /*[this->NumConnections]*/
  };

/**
 */
struct _FaceNeighbor_a
  {
    LgIndex_t                RefCount;
    Boolean_t                IsValid;
    ZoneType_e               ZoneType;
    EntIndex_t               FacesPerElement; /* redundent: fast macro access */
    LgIndex_t                IMax;
    LgIndex_t                JMax;
    LgIndex_t                KMax;
    FaceNeighborMode_e       Mode;
    struct _AutoGenFaceNbr_a *AutoGen; /*NULL for ordered data*/
    struct _UserDefFaceNbr_a *UserDef; /*NULL if not supplied*/
  };


#define FaceNeighborGetFacesPerElem(FaceNeighbor) ((FaceNeighbor)->FacesPerElement)
#define GetAutoFNNumBoundaryFaces(FaceNeighbor)   ((FaceNeighbor)->AutoGen->NumBoundaryFaces)
#define GetAutoFNNumCollapsedFaces(FaceNeighbor)  ((FaceNeighbor)->AutoGen->NumCollapsedFaces)
#define GetAutoFNCellList(FaceNeighbor)           ((FaceNeighbor)->AutoGen->CellList)
                                                  
#define IsUserFNAvailable(FaceNeighbor)           ((FaceNeighbor)->UserDef != NULL)
#define GetUserFNFaceToCellListMap(FaceNeighbor)  ((FaceNeighbor)->UserDef->FaceToCellListMap)
#define GetUserFNCellList(FaceNeighbor)           ((FaceNeighbor)->UserDef->CellList)
#define GetUserFNZoneList(FaceNeighbor)           ((FaceNeighbor)->UserDef->ZoneList)


/**
 * Face neighbor assignment context is used to collect face neighbor
 * assignments so that user defined face neighbors can be put into a more
 * compressed form once all neighbors have been delivered.
 */
typedef struct _FaceNeighborAssignCtx_s
  {
    Boolean_t     IsOk;
    LgIndex_t     NumBoundaryFaces;
    LgIndex_t     NumMapEntries;
    DataSet_s    *DataSet;
    EntIndex_t    Zone;
    ArrayList_pa  CellList;
    ArrayList_pa  ZoneList;
  } FaceNeighborAssignCtx_s;


/**
 */
typedef struct _FaceNeighborAssignCtx_s  *FaceNeighborAssignCtx_pa;

/**
 */
EXTERN EntIndex_t GetNodesPerElementFace(ZoneType_e ZoneType);
EXTERN EntIndex_t GetFacesPerElement(ZoneType_e ZoneType,
                                     LgIndex_t  IMax,
                                     LgIndex_t  JMax,
                                     LgIndex_t  KMax);
EXTERN CollapsedStatus_e GetSurfaceCellCollapsedStatus(const CZInfo_s                                 *CZInfo,
                                                       const CZData_s                                 *CZData,
                                                       tecplot::kernel::SubElemValueProducerInterface *SubElemValueProducer);
EXTERN CollapsedStatus_e GetSurfaceCellCollapsedStatus(const CZInfo_s *CZInfo,
                                                       const CZData_s *CZData,
                                                       LgIndex_t       I1,
                                                       LgIndex_t       I2,
                                                       LgIndex_t       I3,
                                                       LgIndex_t       I4);
EXTERN CollapsedStatus_e GetSurfaceCellLogicalCollapsedStatus(ZoneType_e ZoneType,
                                                              LgIndex_t  I1,
                                                              LgIndex_t  I2,
                                                              LgIndex_t  I3,
                                                              LgIndex_t  I4);
EXTERN CollapsedStatus_e GetSurfEdgeOrVolFaceLogicalCollapsedStatus(NodeMap_pa NodeMap,
                                                                    LgIndex_t  Element,
                                                                    EntIndex_t Face);
EXTERN Boolean_t IsFEElementLogicallyCollapsed(NodeMap_pa NodeMap,
                                               LgIndex_t  Element);
EXTERN Boolean_t FaceNeighborAllocAutoGen(FaceNeighbor_pa FaceNeighbor);
EXTERN void FaceNeighborUpdateAutoGenBoundaryCount(FaceNeighbor_pa FaceNeighbor);
EXTERN Boolean_t UpdateFaceNeighbor(Boolean_t                          DoInterruptChecking,
                                    FaceNeighbor_pa                    FaceNeighbor,
                                    NodeMap_pa                         NodeMap,
                                    tecplot::strutil::TranslatedString TargetMsg);
EXTERN Boolean_t IsValidFaceNeighbor(FaceNeighbor_pa FaceNeighbor);
EXTERN void InvalidateFaceNeighbor(FaceNeighbor_pa FaceNeighbor);
EXTERN void ValidateFaceNeighbor(FaceNeighbor_pa FaceNeighbor);
EXTERN FaceNeighbor_pa FaceNeighborAlloc(FaceNeighborMode_e Mode,
                                         ZoneType_e         ZoneType,
                                         LgIndex_t          IMax,
                                         LgIndex_t          JMax,
                                         LgIndex_t          KMax);
EXTERN void FaceNeighborDealloc(FaceNeighbor_pa *FaceNeighbor);
EXTERN FaceNeighbor_pa FaceNeighborCopy(FaceNeighbor_pa FaceNeighbor,
                                        Boolean_t       CopyFaceBndryConns);

#if defined USE_MACROS_FOR_FUNCTIONS
#  define GetAutoGenFNElement   GetAutoGenFNElement_MACRO
#  define SetAutoGenFNElement   SetAutoGenFNElement_MACRO
#else
#  define GetAutoGenFNElement   GetAutoGenFNElement_FUNC
#  define SetAutoGenFNElement   SetAutoGenFNElement_FUNC
#endif

#if !defined USE_MACROS_FOR_FUNCTIONS
  EXTERN LgIndex_t GetAutoGenFNElement_FUNC(FaceNeighbor_pa FaceNeighbor,
                                            LgIndex_t       Element,
                                            LgIndex_t       Face);
  EXTERN void SetAutoGenFNElement_FUNC(FaceNeighbor_pa FaceNeighbor,
                                       LgIndex_t       Element,
                                       LgIndex_t       Face,
                                       LgIndex_t       NeighboringElement);
#endif


#define GetAutoGenFNElement_MACRO(FaceNeighbor, Element, Face) \
            ((FaceNeighbor)->AutoGen->CellList[(Element)*(FaceNeighbor)->FacesPerElement + (Face)])
#define SetAutoGenFNElement_MACRO(FaceNeighbor, Element, Face, NeighboringElement) \
            ((FaceNeighbor)->AutoGen->CellList[(Element)*(FaceNeighbor)->FacesPerElement + (Face)] = (NeighboringElement))


/**
 * Gets the "perfect" face neighbor for the specified element's face from the
 * auto generated or user defined face neighbors. The "perfect" face neighbor
 * is one that is either an auto generated face neighbor or a user defined face
 * neighbor that conforms to the same rules: i.e. face neighbors share the same
 * nodes and are reciprocal with regard to being neighbors of one another.
 */
#define FaceNeighborGetPerfectNeighbor(FaceNeighbor, \
                                       Element, \
                                       Face, \
                                       PerfectNeighbor_Addr) \
        { \
          LgIndex_t BaseMapIndex = (Element)*(FaceNeighbor)->FacesPerElement + (Face); \
          if ((FaceNeighbor)->UserDef != NULL) \
            { \
              if ((FaceNeighbor)->AutoGen != NULL) \
                { \
                  *(PerfectNeighbor_Addr) = (FaceNeighbor)->AutoGen->CellList[BaseMapIndex]; \
                  if (*(PerfectNeighbor_Addr) == NO_NEIGHBORING_ELEMENT) \
                    { \
                      LgIndex_t StartOffset = (FaceNeighbor)->UserDef->FaceToCellListMap[BaseMapIndex]; \
                      LgIndex_t EndOffset   = (FaceNeighbor)->UserDef->FaceToCellListMap[BaseMapIndex + 1]; \
                      if (EndOffset - StartOffset == 1 && \
                          ((FaceNeighbor)->UserDef->IsPerfectNeighbor == NULL || \
                           InSet((FaceNeighbor)->UserDef->IsPerfectNeighbor, StartOffset))) \
                        *(PerfectNeighbor_Addr) = (FaceNeighbor)->UserDef->CellList[StartOffset]; \
                      else \
                        *(PerfectNeighbor_Addr) = NO_NEIGHBORING_ELEMENT; \
                    } \
                } \
              else \
                *(PerfectNeighbor_Addr) = NO_NEIGHBORING_ELEMENT; \
            } \
          else \
            { \
              if ((FaceNeighbor)->AutoGen != NULL) \
                *(PerfectNeighbor_Addr) = (FaceNeighbor)->AutoGen->CellList[BaseMapIndex]; \
              else \
                *(PerfectNeighbor_Addr) = NO_NEIGHBORING_ELEMENT; \
            } \
        }

/**
 * Indicates if the element's face has a neighboring element. With regard to
 * this function, the face is considered to have a neighbor if and only if it:
 *
 *   - has an auto generated neighbor
 *   - has a single, completely obscured, user defined neighbor
 */
EXTERN Boolean_t HasOnlyOneLocalFaceNeighbor(FaceNeighbor_pa FaceNeighbor,
                                             LgIndex_t       Element,
                                             LgIndex_t       Face,
                                             LgIndex_t       *NeighboringElement);
/**
 */
EXTERN LgIndex_t GetLogicalOrderedNeighbor(LgIndex_t NumIPts,
                                           LgIndex_t NumJPts,
                                           LgIndex_t NumKPts,
                                           LgIndex_t Element,
                                           LgIndex_t Face);

#if defined ALLOW_USERDEF_NO_NEIGHBORING_ELEMENT
/**
 */
EXTERN Boolean_t IsUserDefFaceNeighborBoundary(FaceNeighbor_pa FaceNeighbor,
                                               LgIndex_t       Element,
                                               LgIndex_t       Face);
#endif
/**
 * Gets the user defined face neighbor mode.
 */
EXTERN FaceNeighborMode_e FaceNeighborGetMode(FaceNeighbor_pa FaceNeighbor);
   
/**
 */
EXTERN LgIndex_t FaceNeighborGetNumElements(FaceNeighbor_pa FaceNeighbor);

/**
 * Gets the face obscuration information for the specified face based on the
 * face neighbor and logical connectivity.
 *
 * LIMITATION:
 *     Currently this function does not consider blanking within or without the
 *     current zone or COB (see GetFaceObscuration for a function that does
 *     consider blanking). Blanking depends on a setup zone or COB and this
 *     function operates outside of a current zone. Even if we did setup the
 *     zone, we still would not be complete because global face neighbors
 *     can refer to cells from other zones.
 *
 * param ActiveZones
 *     Active set of zones used only for checking that global face neighbors
 *     are active.
 */
EXTERN FaceObscuration_e FaceNeighborGetFaceObscuration(FaceNeighbor_pa FaceNeighbor,
                                                        LgIndex_t       Element,
                                                        LgIndex_t       Face,
                                                        Set_pa          ActiveZones);
/**
 * Gets the number of neighboring elements for the specified element and face.
 */
EXTERN LgIndex_t FaceNeighborGetNumNeighbors(FaceNeighbor_pa  FaceNeighbor,
                                             LgIndex_t        Element,
                                             LgIndex_t        Face,
                                             Boolean_t       *AreNeighborsUserDef);
/**
 * Gets a reference to the neighboring elements and zones for the specified
 * element, face, and neighbor number. This provides direct access to the
 * internal array and should not be modified, Nor can the returned array be
 * expected not to change between calls. Therefore the element and zone members
 * should be copied if persistence of the information is needed.
 *
 * @param FaceNeighbor
 *     Face neighbor pointer.
 * @param Element
 *     Element for which the neighbors are desired.
 * @param Face
 *     Element face for which the face neighbors are desired.
 * @param NeighborElems
 *     Resulting neighboring element list to the specified element's face.
 * @param NeighborZones
 *     Resulting neighboring zone list to the specified element's face. This
 *     will only be non-NULL for global face neighbors.
 * @param OrderedLogicalNeighborElem
 *     This buffer is only used for ordered data's single logical neighbors
 *     where an internal array into the face neighbor structure does not
 *     exists because the logical neighbor is implied. It is ignored for finite
 *     element data (and for some ordered data if user defined face neighbors
 *     exists for this element's face). Since the buffer may or may not be used
 *     the result of this call should be examined through the NeighborElems
 *     array.
 * @param OrderedLogicalNeighborZone
 *     This buffer is only used for ordered data's single logical neighbors
 *     where an internal array into the face neighbor structure does not
 *     exists because the logical neighbor is implied. It is ignored for finite
 *     element data (and for some ordered data if user defined face neighbors
 *     exists for this element's face). Since the buffer may or may not be used
 *     the result of this call should be examined through the NeighborZones
 *     array.
 */
EXTERN void FaceNeighborGetNeighbors(FaceNeighbor_pa    FaceNeighbor,
                                     LgIndex_t          Element,
                                     LgIndex_t          Face,
                                     const LgIndex_t  **NeighborElems,
                                     const EntIndex_t **NeighborZones,
                                     LgIndex_t         *OrderedLogicalNeighborElem,
                                     EntIndex_t        *OrderedLogicalNeighborZone);
/**
 * Gets the neighboring element and zone for the specified element, face, and
 * neighbor number.
 */
EXTERN void FaceNeighborGetNeighbor(FaceNeighbor_pa FaceNeighbor,
                                    LgIndex_t       Element,
                                    LgIndex_t       Face,
                                    LgIndex_t       NeighborNumber,
                                    LgIndex_t       *NeighborElem,
                                    EntIndex_t      *NeighborZone);
/**
 */
EXTERN void FaceNeighborGetSurfaceCellNeighbor(const CZInfo_s *CZInfo,
                                               const CZData_s *CZData,
                                               LgIndex_t       SurfaceCellIndex,
                                               int             PlaneOrFace,
                                               SmInteger_t     Edge,
                                               LgIndex_t      *NeighborSurfaceCellElem,
                                               EntIndex_t     *NeighborSurfaceCellZone);
/**
 */
EXTERN FaceObscuration_e GetFaceObscuration(const CZInfo_s *CZInfo,
                                            const CZData_s *CZData,
                                            LgIndex_t       Element,
                                            LgIndex_t       Face,
                                            Boolean_t       ConsiderValueBlanking,
                                            Boolean_t       ConsiderIJKBlanking,
                                            Boolean_t       ConsiderDepthBlanking);

/**
 */
EXTERN Boolean_t IsFaceNeighborAssignContextOk(FaceNeighborAssignCtx_pa FaceNeighborAssignCtx);

/**
 * Clears any previous user defined face neighbor assignments and opens a new
 * face neighbor assignment context. The context must be closed with a call to
 * FaceNeighborEndAssign when all face neighbor assignments have been
 * delivered.
 */
EXTERN FaceNeighborAssignCtx_pa FaceNeighborUserDefBeginAssign(DataSet_s    *DataSet,
                                                               ArrayList_pa  ZoneStyleStateList,
                                                               EntIndex_t    Zone);

/**
 * Assigns the user defined face neighbors within an open face neighbor
 * assignment context for the specified element and face.
 */
EXTERN Boolean_t FaceNeighborUserDefAssign(FaceNeighborAssignCtx_pa  FaceNeighborAssignCtx,
                                           LgIndex_t                 Element,
                                           LgIndex_t                 Face,
                                           Boolean_t                 NeighborsCompletelyObscure,
                                           LgIndex_t                 NumNeighbors,
                                           LgIndex_t                *NeighborElems,
                                           EntIndex_t               *NeighborZones);
/**
 * Closes the open face neighbor assignment context and packs the assignments
 * into an efficient storage within Tecplot.
 */
EXTERN Boolean_t FaceNeighborUserDefEndAssign(FaceNeighborAssignCtx_pa *FaceNeighborAssignCtx);

/**
 */
EXTERN Boolean_t VerifyUserDefFNIntegrity(DataSet_s  *DataSet,
                                          EntIndex_t Zone);

/**
 * Adjusts the zone numbers (if applicable) of the face neighbors following the
 * zone deleted or inserted by the specified amount and invalidates the face
 * zone neighbor if it references the adjusted zone.
 *
 * @param DataSet
 *     Data set needing adjustment.
 * @param ZoneToCheck
 *     Zone number to check.
 * @param ZoneAdjusted
 *     Zone number that was either inserted, deleted, or realloced.
 * @param ZoneAdjustment
 *     Numerical adjustment to apply to all face neighbor zone references that
 *     need adjustment:
 *       -1 : ZoneAdjustment zone was deleted
 *        0 : ZoneAdjustment zone was adjusted without changing position
 *       +1 : ZoneAdjustment zone was inserted
 *
 * @return
 *     TRUE if successful, FALSE otherwise.
 */
EXTERN Boolean_t FaceNeighborAdjustUserDefAfterZoneAdjustment(DataSet_s  *DataSet,
                                                              EntIndex_t  ZoneToCheck,
                                                              EntIndex_t  ZoneAdjusted,
                                                              EntIndex_t  ZoneAdjustment);


/**
 * Function to determine a cell's neighbor.  It calls FaceNeighborGetSurfaceCellNeighbor() 
 * for classic zones.
 */

EXTERN void GetSurfaceCellNeighbor(const CZInfo_s                                 *CZInfo,
                                   const CZData_s                                 *CZData,
                                   LgIndex_t                                       SurfaceCellIndex,
                                   tecplot::kernel::SubElemValueProducerInterface *SubElemValueProducer,
                                   ElemFaceOffset_t                                PlaneOrFaceOffset,
                                   ElemFaceOffset_t                                Edge,
                                   LgIndex_t                                      *NeighborSurfaceCellElem,
                                   EntIndex_t                                     *NeighborSurfaceCellZone);
#endif
