/*
******************************************************************
******************************************************************
*******                                                   ********
******  (C) 1988-2008 Tecplot, Inc.                        *******
*******                                                   ********
******************************************************************
******************************************************************
*/


#if !defined Q_UNICODE_H_
# define Q_UNICODE_H_

#if defined EXTERN
  #undef EXTERN
#endif
#if defined Q_UNICODEMODULE
  #define EXTERN
#else
  #define EXTERN extern
#endif

namespace tecplot { namespace strutil {

      // functions
      Boolean_t IsValidUtf8LeadByte(Byte_t ch);
      Boolean_t IsValidUtf8ContinuingByte(Byte_t ch);
      Boolean_t IsValidUtf8Byte(Byte_t ch);

      Boolean_t IsValidUtf8String(const char *str);
      Boolean_t ShouldConvertWideStringToUtf8String(const wchar_t *str);
      void InitTranslatedStrings();
      void CleanUpTranslatedStrings();

      Boolean_t IsNullOrZeroLengthString(const char *S);
      Boolean_t IsNullOrZeroLengthString(tecplot::strutil::TranslatedString TS);

      Boolean_t IsEmptyString(const char *S);
      Boolean_t IsEmptyString(tecplot::strutil::TranslatedString S);
      Boolean_t IsEmptyString(const wchar_t* S);

#if defined MSWIN
  
      std::string  LookUpTranslation(std::string& strEnglish);
      void MsWinInitTranslatedStrings();

      std::string    WStringToString(std::wstring str);
      std::wstring   StringToWString(std::string str);

      std::wstring   MultiByteToWideChar(const char *Utf8Str,
                                         unsigned int    CodePage);

      std::string    WideCharToMultiByte(const wchar_t *WideStr,
                                         unsigned int    CodePage);

      // file ops
#if defined TECPLOTKERNEL
/* CORE SOURCE CODE REMOVED */
#endif

      // Conversion
      std::string    WideCharToUtf8(const wchar_t* str);
      std::wstring   Utf8ToWideChar(const char *str);
      char *getenv(const char *str);

#endif 

}}

#endif 
