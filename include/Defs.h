// #define USE_ORBFEATURES

#ifndef DEFS_SP_ORBSLAM
#define DEFS_SP_ORBSLAM

#ifdef DEBUG_PRINTF
#define DBG_PRINTF printf
#else
#define DBG_PRINTF(...)
#endif

const bool SP_USE_CUDA = false;
// #define USE_DBOW2
// #define USE_BINARY_DESCRIPTORS 
#define DBOW_LEVELS 6 // Banafshe chenged 0 to 6
#define ENABLE_SUBBLOCKS_KEY_EXTRACTION

// Added by B.B: to use my implementation of SuperPoint
#define USE_SELM_EXTRACTOR

#endif






