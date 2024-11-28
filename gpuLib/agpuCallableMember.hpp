#pragma once

#if defined(__CUDACC__) || defined(__HIPCC__)
#define AGPU_CALLABLE_MEMBER __host__ __device__
#else
#define AGPU_CALLABLE_MEMBER
#endif

