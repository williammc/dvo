// Description: import/export macros for creating DLLS with MSVC
// Use SM_<module>_API macro to declare it.
// Sparate macros for each DLL required.
#pragma once
#pragma warning(disable:4251) // needs to have dll-interface for class members
#ifdef _MSC_VER  // Microsoft compiler:
  #ifdef SM_SHARED_LIBS
    #ifdef dvo_EXPORTS
      #define DVO_API __declspec(dllexport)
    #else
      #define DVO_API __declspec(dllimport)
    #endif
  #else
    #define DVO_API
  #endif

#else  // not MSVC
  #define DVO_API
#endif