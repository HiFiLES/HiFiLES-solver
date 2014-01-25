#if !defined STDAFX_H_
# define STDAFX_H_

/*
******************************************************************
******************************************************************
*******                                                   ********
******  (C) 1988-2008 Tecplot, Inc.                        *******
*******                                                   ********
******************************************************************
******************************************************************
*/
/*
 * stdafx.h : include file for standard system include files,
 *  or project specific include files that are used frequently, but
 *      are changed infrequently
 * Used for Windows only
 */
#if defined _WIN32

  /*
   * Set NDEBUG before including "custafx.h" since that file may
   * use NDEBUG.  (In fact, for SmartHeap builds this is the case.)
   * CAM 2007-04-11
   *
   * Previous comment: "Note that _DEBUG is defined by the Windows compiler
   * if any of the multi-threaded DLL runtime libraries are used."
   */
  #if !defined _DEBUG
    #if !defined NDEBUG
      #define NDEBUG
    #endif
  #endif

  #if defined TECPLOTKERNEL
    #include "custafx.h"
  #endif /* TECPLOTKERNEL */
  
  #if !defined MSWIN  
    #define MSWIN
  #endif
  
    // Turn of smart heap for win 64 builds
  #if defined _WIN64 && defined USE_SMARTHEAP
    #undef USE_SMARTHEAP
  #endif

  #if defined USE_SMARTHEAP
    #pragma message("Using SMARTHEAP...")
    /* SmartHeap must be first so its libraries are scanned first for malloc/free/etc. */
    //#define USE_SMARTHEAP_SMP_VERSION_8_1
    #define USE_SMARTHEAP_VERSION_8_1
    #if defined _WIN64
      #error "SmartHeap not available for Win64"
    #else
      #ifdef _DEBUG
        #error "Don't build debug versions with SmartHeap"
      #else
        #if defined USE_SMARTHEAP_SMP_VERSION_8_1
          #include "\\cedar\workgroups\marketing\jim\SmartHeapSMP\windows\include\smrtheap.h"
          #pragma comment(lib, "\\\\cedar\\workgroups\\marketing\\jim\\SmartHeapSMP\\windows\\msvc\\shdsmpmt.lib")
        #elif defined USE_SMARTHEAP_VERSION_8_1
          #include "K:\development\tecplot\libs\SmartHeap8.1\Sh81winb\include\smrtheap.h"
          #pragma comment(lib, "K:\\development\\tecplot\\libs\\SmartHeap8.1\\Sh81winb\\msvc\\shdw32mt.lib")
        #elif defined USE_SMARTHEAP_VERSION_8_0
          #include "\\cedar\workgroups\development\tecplot\Builds\smartheap\8.0.0\include\smrtheap.h"
          #pragma comment(lib, "\\\\cedar\\workgroups\\development\\tecplot\\Builds\\smartheap\\8.0.0\\msvc\\shdw32mt.lib")
        #else
          #include "\\cedar\workgroups\development\tecplot\Builds\smartheap\6.0.3\include\smrtheap.h"
          #pragma comment(lib, "\\\\cedar\\workgroups\\development\\tecplot\\Builds\\smartheap\\6.0.3\\msvc\\shdw32mt.lib")
        #endif
      #endif
    #endif
  #endif
  
  #define ENGLISH_ONLY // remove to support non-english dll's
  
  #if !defined WINVER
    #define WINVER 0x0500
  #endif
  
  #if defined TECPLOTKERNEL
    #if defined CHECKED_BUILD || defined _DEBUG
      /* This will switch to the "safe" versions of
         the c runtimes which check for overflows, etc. */
      #define _CRT_SECURE_CPP_OVERLOAD_STANDARD_NAMES 1
    #else
      /* Use regular c runtimes (which are deprecated in VS2005), 
         but don't warn about it */
      #define _CRT_SECURE_NO_DEPRECATE
    #endif
  #endif /* TECPLOTKERNEL */
  
  /*  Windows builds are UNICODE */
  #pragma warning(disable : 4786) /* truncated identifiers in debug symbol table. */
  #pragma warning(disable : 4996) /* deprecated functions */

  #if defined TECPLOTKERNEL
    #define UNICODE
    #define _UNICODE

    #if !defined _AFXDLL
      #define _AFXDLL
    #endif
  
    #if defined _M_IX86
      #pragma comment(linker,"/manifestdependency:\"type='win32' name='Microsoft.Windows.Common-Controls' version='6.0.0.0' processorArchitecture='x86' publicKeyToken='6595b64144ccf1df' language='*'\"")
    #elif defined _M_IA64
      #pragma comment(linker,"/manifestdependency:\"type='win32' name='Microsoft.Windows.Common-Controls' version='6.0.0.0' processorArchitecture='ia64' publicKeyToken='6595b64144ccf1df' language='*'\"")
    #elif defined _M_X64
      #pragma comment(linker,"/manifestdependency:\"type='win32' name='Microsoft.Windows.Common-Controls' version='6.0.0.0' processorArchitecture='amd64' publicKeyToken='6595b64144ccf1df' language='*'\"")
    #else
      #pragma comment(linker,"/manifestdependency:\"type='win32' name='Microsoft.Windows.Common-Controls' version='6.0.0.0' processorArchitecture='*' publicKeyToken='6595b64144ccf1df' language='*'\"")
    #endif
  

    #if defined _WIN64
      #if !defined _M_IA64 && !defined _M_AMD64
        #error "UNKNOWN 64-BIT PLATFORM"
      #endif
    #endif

    /* TECPLOT preprocessor defines */
    #if !defined MSWIN
      #define MSWIN
    #endif
    #define THREED
    

    #include <afxwin.h>         /* MFC core and standard components */
    #include <afxext.h>         /* MFC extensions */
    /* #include <afxdao.h> */
    #include <afxinet.h>
    #include <afxmt.h> /* multi-threaded stuff */
    #include <afxtempl.h> /* templated collection classes */
    #include <uxtheme.h>
    #include <tmschema.h>

    #ifndef _AFX_NO_AFXCMN_SUPPORT
    #include <afxcmn.h>   /* MFC support for Windows Common Controls */
    #endif /* _AFX_NO_AFXCMN_SUPPORT */


    #ifndef _AFX
    #error MFC is not defined as far as Tecplot is concerned.
    #endif

    #include <cderr.h> /* comm dialog error codes */
    #include <shlobj.h>
    #include <winver.h>
    #include <mbstring.h>

    #include <Iphlpapi.h>
    #include <io.h>

    #include "MASTER.h"

    /* To help us keep the source code clean,
       enable the unreferenced function warning. */
    #pragma warning(3 : 4505)  /* Unreferenced function */
    

    /*
     * These are necessary, otherwise you get massive warnings. CAM 03/17/2004
     */
    #pragma warning (disable : 4244) /* conversion: 1856 warnings */
    #pragma warning (disable : 4100) /* unreferenced formal parameter: 331 warnings */

    /*
     * This one doesn't appear to do anything in debug, but supresses
     * countless warnings in release builds. CAM 03/05/2004
     */
    #pragma warning (disable : 4711) /* inline function */

    /*
     * It would like to turn this one back on but it created 218 warnings, more
     * that I had time to deal with. CAM 03/05/2004
     */
    #pragma warning (disable : 4701) /* variable 'may' be used without having been initialized */  

    /*
     * C++ exception specification ignored except to
     * indicate a function is not __declspec(nothrow)
     */
    #pragma warning(disable: 4290)

    /* linker settings now in stdafx.cpp */

  #else /* !TECPLOTKERNEL */
    #define AfxIsValidAddress(ptr,bb) ((ptr)!=NULL)
  #endif

  /* 64-bit adjustments */
  #if defined _M_IA64 || defined _M_AMD64
    #define WININT  INT_PTR
    #define WINUINT UINT_PTR
  #else
    #define WININT  int
    #define WINUINT UINT
  #endif
  
  #define WINCALLBACK CALLBACK
  
  #if defined TECPLOTKERNEL
    /* message tracing */
    #if defined (NDEBUG)
      #define TRACE_WINDOWS_MESSAGE(m,w,l)
    #else
      extern void InternalWindowsTraceMessage(UINT m, WPARAM w, LPARAM l);
      #define TRACE_WINDOWS_MESSAGE(m,w,l) InternalWindowsTraceMessage(m,w,l)
    #endif
  #endif /* TECPLOTKERNEL */
#endif /* _WIN32 */


#endif /* STDAFX_H_ */
