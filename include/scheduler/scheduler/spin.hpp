#pragma once

// Low-latency spin hint (avoid context switch, reduce power)
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
# include <immintrin.h>
# define SPIN_HINT() _mm_pause()
#elif defined(__aarch64__) || defined(__arm__)
# define SPIN_HINT() __asm__ volatile("yield" ::: "memory")
#else
# define SPIN_HINT() ((void)0)
#endif
