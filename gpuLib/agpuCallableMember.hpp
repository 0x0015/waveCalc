#pragma once

#if defined(__CUDACC__) || defined(__HIPCC__)
#define AGPU_CALLABLE_MEMBER __host__ __device__
#define DETECTED_AGPU_COMPATABLE_COMPILER
#else
#define AGPU_CALLABLE_MEMBER
#endif

