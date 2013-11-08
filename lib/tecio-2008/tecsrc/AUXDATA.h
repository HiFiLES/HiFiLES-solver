/* 
 *****************************************************************
 *****************************************************************
 *******                                                  ********
 ****** Copyright (C) 1988-2008 Tecplot, Inc.              *******
 *******                                                  ********
 *****************************************************************
 *****************************************************************
 */
#if !defined AUXDATA_h
#define AUXDATA_h

#if defined EXTERN
#  undef EXTERN
#endif
#if defined AUXDATAMODULE
#  define EXTERN
#else
#  define EXTERN extern
#endif

/**
 */
EXTERN Boolean_t AuxDataIsValidNameChar(char      Char,
                                        Boolean_t IsLeadChar);
/**
 */
EXTERN Boolean_t AuxDataIsValidName(const char *Name);

/**
 */
EXTERN AuxData_pa AuxDataAlloc(void);

/**
 */
EXTERN void AuxDataDealloc(AuxData_pa *AuxData);

/**
 */
EXTERN Boolean_t AuxDataItemDestructor(void       *ItemRef,
                                       ArbParam_t  ClientData);
/**
 */
EXTERN AuxData_pa AuxDataCopy(AuxData_pa AuxData,
                              Boolean_t  ConsiderRetain);

/**
 */
EXTERN LgIndex_t AuxDataGetNumItems(AuxData_pa AuxData);

/**
 */
EXTERN Boolean_t AuxDataGetItemIndex(AuxData_pa AuxData,
                                     const char *Name,
                                     LgIndex_t  *ItemIndex);
/**
 */
EXTERN void AuxDataGetItemByIndex(AuxData_pa    AuxData,
                                  LgIndex_t     Index,
                                  const char    **Name,
                                  ArbParam_t    *Value,
                                  AuxDataType_e *Type,
                                  Boolean_t     *Retain);

/**
 */
EXTERN Boolean_t AuxDataGetItemByName(AuxData_pa    AuxData,
                                      const char    *Name,
                                      ArbParam_t    *Value,
                                      AuxDataType_e *Type,
                                      Boolean_t     *Retain);

/**
 */
EXTERN Boolean_t AuxDataGetBooleanItemByName(AuxData_pa     AuxData,
                                             const char    *Name,
                                             Boolean_t     *Value,
                                             AuxDataType_e *Type,
                                             Boolean_t     *Retain);

/**
 */
EXTERN Boolean_t AuxDataSetItem(AuxData_pa    AuxData,
                                const char    *Name,
                                ArbParam_t    Value,
                                AuxDataType_e Type,
                                Boolean_t     Retain);

/**
 */
EXTERN Boolean_t AuxDataDeleteItemByName(AuxData_pa AuxData,
                                         const char *Name);

/**
 */
EXTERN Boolean_t AuxDataAppendItems(AuxData_pa TargetAuxData,
                                    AuxData_pa SourceAuxData);
/**
 */
EXTERN void AuxDataDeleteItemByIndex(AuxData_pa AuxData,
                                     LgIndex_t  Index);

#if defined TECPLOTKERNEL
/* CORE SOURCE CODE REMOVED */
#endif /* TECPLOTKERNEL */

#endif /* !defined AUXDATA_h */
