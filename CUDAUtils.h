#pragma once
#include <cstdint>
#include <string>
#include <sstream>
#include <iomanip>
#include <cmath>
#include <algorithm> // std::min

#if defined(_MSC_VER) && !defined(__CUDA_ARCH__)
  #include <intrin.h> // _udiv128, _addcarry_u64, _subborrow_u64
#endif

// ===================== 256-bit helpers (HOST) =====================

// a + b(64-bit) -> out (256-bit), correct carry spread
__host__ __forceinline__ void add256_u64(const uint64_t a[4], uint64_t b, uint64_t out[4]) {
#if defined(_MSC_VER) && !defined(__CUDA_ARCH__)
    unsigned char c = 0;
    c = _addcarry_u64(0, a[0], b,    &out[0]);
    c = _addcarry_u64(c, a[1], 0ull, &out[1]);
    c = _addcarry_u64(c, a[2], 0ull, &out[2]);
    _addcarry_u64(c, a[3], 0ull,     &out[3]);
#elif defined(__SIZEOF_INT128__) && !defined(__CUDA_ARCH__)
    unsigned __int128 s = (unsigned __int128)a[0] + (unsigned __int128)b;
    out[0] = (uint64_t)s;
    uint64_t carry = (uint64_t)(s >> 64);
    for (int i = 1; i < 4; ++i) {
        s = (unsigned __int128)a[i] + (unsigned __int128)carry;
        out[i] = (uint64_t)s;
        carry = (uint64_t)(s >> 64);
    }
#else
    // Pure 64-bit fallback
    uint64_t carry = b;
    for (int i = 0; i < 4; ++i) {
        const uint64_t ai = a[i];
        const uint64_t s  = ai + carry;
        out[i] = s;
        carry = (carry ? (s <= ai ? 1ull : 0ull) : 0ull);
    }
#endif
}

// a + b(256-bit) -> out (256-bit)
__host__ __forceinline__ void add256(const uint64_t a[4], const uint64_t b[4], uint64_t out[4]) {
#if defined(_MSC_VER) && !defined(__CUDA_ARCH__)
    unsigned char c = 0;
    c = _addcarry_u64(0, a[0], b[0], &out[0]);
    c = _addcarry_u64(c, a[1], b[1], &out[1]);
    c = _addcarry_u64(c, a[2], b[2], &out[2]);
    _addcarry_u64(c, a[3], b[3],     &out[3]);
#elif defined(__SIZEOF_INT128__) && !defined(__CUDA_ARCH__)
    unsigned __int128 s = (unsigned __int128)a[0] + (unsigned __int128)b[0];
    out[0] = (uint64_t)s;
    uint64_t carry = (uint64_t)(s >> 64);
    for (int i = 1; i < 4; ++i) {
        s = (unsigned __int128)a[i] + (unsigned __int128)b[i] + (unsigned __int128)carry;
        out[i] = (uint64_t)s;
        carry = (uint64_t)(s >> 64);
    }
#else
    uint64_t carry = 0ull;
    for (int i = 0; i < 4; ++i) {
        const uint64_t ai = a[i], bi = b[i];
        const uint64_t s  = ai + bi;
        const uint64_t c1 = (s < ai) ? 1ull : 0ull;                   // carry of ai+bi
        const uint64_t s2 = s + carry;
        const uint64_t c2 = (carry ? (s2 <= s ? 1ull : 0ull) : 0ull); // carry of +carry
        out[i] = s2;
        carry  = c1 | c2;
    }
#endif
}

// a - b(256-bit) -> out (256-bit)  (correct borrow)
__host__ __forceinline__ void sub256(const uint64_t a[4], const uint64_t b[4], uint64_t out[4]) {
#if defined(_MSC_VER) && !defined(__CUDA_ARCH__)
    unsigned char br = 0;
    br = _subborrow_u64(0, a[0], b[0], &out[0]);
    br = _subborrow_u64(br, a[1], b[1], &out[1]);
    br = _subborrow_u64(br, a[2], b[2], &out[2]);
    _subborrow_u64(br, a[3], b[3], &out[3]);
#elif defined(__SIZEOF_INT128__) && !defined(__CUDA_ARCH__)
    uint64_t borrow = 0ull;
    for (int i = 0; i < 4; ++i) {
        const unsigned __int128 bi = (unsigned __int128)b[i] + (unsigned __int128)borrow;
        const unsigned __int128 ai = (unsigned __int128)a[i];
        const unsigned __int128 r  = ai - bi;
        out[i]  = (uint64_t)r;
        borrow  = (ai < bi) ? 1ull : 0ull;
    }
#else
    uint64_t borrow = 0ull;
    for (int i = 0; i < 4; ++i) {
        const uint64_t ai = a[i];
        const uint64_t bi = b[i];
        const uint64_t t  = ai - bi;
        const uint64_t b0 = (ai < bi) ? 1ull : 0ull;
        const uint64_t r  = t - borrow;
        const uint64_t b1 = (t < borrow) ? 1ull : 0ull;
        out[i] = r;
        borrow = b0 | b1;
    }
#endif
}

__host__ __forceinline__ void inc256(uint64_t a[4], uint64_t inc) {
    // a += inc (in place)
    const uint64_t s0 = a[0] + inc;
    uint64_t carry = (s0 < a[0]) ? 1ull : 0ull;
    a[0] = s0;
    for (int i = 1; i < 4 && carry; ++i) {
        const uint64_t ai = a[i];
        const uint64_t s  = ai + 1ull;
        a[i] = s;
        carry = (s <= ai) ? 1ull : 0ull;
    }
}

// ===== Helpers power-of-two/alignment (HOST) =====
__host__ __forceinline__ bool is_zero_256_host(const uint64_t a[4]) {
    return ((a[0] | a[1] | a[2] | a[3]) == 0ull);
}

__host__ __forceinline__ void sub256_u64_host(const uint64_t a[4], uint64_t dec, uint64_t out[4]) {
    uint64_t b[4] = { dec, 0ull, 0ull, 0ull };
    sub256(a, b, out);
}

__host__ __forceinline__ void sub1_256_host(const uint64_t a[4], uint64_t out[4]) {
    sub256_u64_host(a, 1ull, out);
}

__host__ __forceinline__ bool is_power_of_two_256_host(const uint64_t a[4]) {
    if (is_zero_256_host(a)) return false;
    uint64_t am1[4];
    sub256_u64_host(a, 1ull, am1);
    // (a & (a-1)) == 0  en 256 bits (LE limbs)
    return ((a[0] & am1[0]) | (a[1] & am1[1]) | (a[2] & am1[2]) | (a[3] & am1[3])) == 0ull;
}

__host__ __forceinline__ bool is_aligned_to_len_256_host(const uint64_t start[4], const uint64_t len[4]) {
    uint64_t lenm1[4];
    sub256_u64_host(len, 1ull, lenm1);
    return ((start[0] & lenm1[0]) | (start[1] & lenm1[1]) | (start[2] & lenm1[2]) | (start[3] & lenm1[3])) == 0ull;
}

// ===================== 256-bit divmod (HOST + DEVICE) =====================
// value (256b) = quotient * divisor + remainder, with 0 <= remainder < divisor
__host__ __device__ inline void divmod_256_by_u64(const uint64_t value[4], uint64_t divisor,
                                                  uint64_t quotient[4], uint64_t &remainder) {
#ifdef __CUDA_ARCH__
    // Device: bit-by-bit division (no __uint128_t on device)
    remainder = 0ull;
    for (int limb = 3; limb >= 0; --limb) {
        uint64_t qword = 0ull;
        const uint64_t v = value[limb];
#pragma unroll
        for (int bit = 63; bit >= 0; --bit) {
            remainder = (remainder << 1) | ((v >> bit) & 1ull);
            if (remainder >= divisor) {
                remainder -= divisor;
                qword |= (1ull << bit);
            }
        }
        quotient[limb] = qword;
    }
#else
    // Host: __uint128_t or _udiv128 (MSVC)
    remainder = 0ull;
    for (int i = 3; i >= 0; --i) {
    #if defined(_MSC_VER)
        unsigned __int64 rem = (unsigned __int64)remainder;
        unsigned __int64 q   = _udiv128(rem,
                                        (unsigned __int64)value[i],
                                        (unsigned __int64)divisor,
                                        (unsigned __int64*)&rem);
        quotient[i] = (uint64_t)q;
        remainder   = (uint64_t)rem;
    #else
        __uint128_t cur = (((__uint128_t)remainder) << 64) | value[i];
        quotient[i] = (uint64_t)(cur / divisor);
        remainder   = (uint64_t)(cur % divisor);
    #endif
    }
#endif
}

// ===================== small host utilities =====================

// Parse 0..64 hex digits (optional "0x") into 256-bit LE (4x64)
inline bool hexToLE64(const std::string& h_in, uint64_t w[4]) {
    std::string h = h_in;
    if (h.size() >= 2 && (h[0] == '0') && (h[1] == 'x' || h[1] == 'X')) h = h.substr(2);
    if (h.empty() || h.find_first_not_of("0123456789abcdefABCDEF") != std::string::npos) return false;
    if (h.size() > 64) return false;

    w[0] = w[1] = w[2] = w[3] = 0ull;

    int pos = (int)h.size();
    for (int limb = 0; limb < 4 && pos > 0; ++limb) {
        const int take = std::min(16, pos);
        std::string part = h.substr(pos - take, take);
        if ((int)part.size() < 16) part = std::string(16 - (int)part.size(), '0') + part;
        try {
            w[limb] = (uint64_t)std::stoull(part, nullptr, 16);
        } catch (...) { return false; }
        pos -= take;
    }
    return true;
}

inline bool hexToHash160(const std::string& h, uint8_t hash160[20]) {
    if (h.size() != 40) return false;
    try {
        for (int i = 0; i < 20; ++i) {
            const std::string byteStr = h.substr(i * 2, 2);
            hash160[i] = (uint8_t)std::stoul(byteStr, nullptr, 16);
        }
    } catch (...) { return false; }
    return true;
}

inline std::string formatHex256(const uint64_t limbs[4]) {
    std::ostringstream oss;
    oss << std::hex << std::uppercase << std::setfill('0');
    for (int i = 3; i >= 0; --i) {
        oss << std::setw(16) << limbs[i];
    }
    return oss.str();
}

inline std::string human_bytes(double bytes) {
    static const char* u[]={"B","KB","MB","GB","TB","PB"};
    int k=0;
    while(bytes>=1024.0 && k<5){ bytes/=1024.0; ++k; }
    std::ostringstream o; o.setf(std::ios::fixed);
    o<<std::setprecision(bytes<10?2:1)<<bytes<<" "<<u[k];
    return o.str();
}

inline long double ld_from_u256(const uint64_t v[4]) {
    return std::ldexp((long double)v[3],192)
         + std::ldexp((long double)v[2],128)
         + std::ldexp((long double)v[1],64)
         + (long double)v[0];
}

inline std::string formatCompressedPubHex(const uint64_t Rx[4], const uint64_t Ry[4]) {
    uint8_t out[33];
    out[0] = (Ry[0] & 1ULL) ? 0x03 : 0x02;
    int off=1;
    for (int limb=3; limb>=0; --limb) {
        const uint64_t v = Rx[limb];
        out[off+0]=(uint8_t)(v>>56); out[off+1]=(uint8_t)(v>>48);
        out[off+2]=(uint8_t)(v>>40); out[off+3]=(uint8_t)(v>>32);
        out[off+4]=(uint8_t)(v>>24); out[off+5]=(uint8_t)(v>>16);
        out[off+6]=(uint8_t)(v>> 8); out[off+7]=(uint8_t)(v>> 0);
        off+=8;
    }
    static const char* hexd="0123456789ABCDEF";
    std::string s; s.resize(66);
    for (int i=0;i<33;++i){ s[2*i]=hexd[(out[i]>>4)&0xF]; s[2*i+1]=hexd[out[i]&0xF]; }
    return s;
}

// ===================== device-side helpers =====================

__device__ __forceinline__ void inc256_device(uint64_t a[4], uint64_t inc) {
    const uint64_t s0 = a[0] + inc;
    uint64_t carry = (s0 < a[0]) ? 1ull : 0ull;
    a[0] = s0;
#pragma unroll
    for (int i = 1; i < 4 && carry; ++i) {
        const uint64_t ai = a[i];
        const uint64_t s  = ai + 1ull;
        a[i] = s;
        carry = (s <= ai) ? 1ull : 0ull;
    }
}

static __device__ __forceinline__ uint32_t load_u32_le(const uint8_t* p) {
    return (uint32_t)p[0]
         | ((uint32_t)p[1] << 8)
         | ((uint32_t)p[2] << 16)
         | ((uint32_t)p[3] << 24);
}

static __device__ __forceinline__ bool hash160_matches_prefix_then_full(
    const uint8_t* __restrict__ h,
    const uint8_t* __restrict__ target,
    const uint32_t target_prefix_le)
{
    if (load_u32_le(h) != target_prefix_le) return false;
#pragma unroll
    for (int k = 4; k < 20; ++k) {
        if (h[k] != target[k]) return false;
    }
    return true;
}

static __device__ __forceinline__ bool hash160_prefix_equals(
    const uint8_t* __restrict__ h,
    uint32_t target_prefix)
{
    return load_u32_le(h) == target_prefix;
}

__device__ __forceinline__ bool ge256_u64(const uint64_t a[4], uint64_t b) {
    if (a[3] != 0ull) return true;
    if (a[2] != 0ull) return true;
    if (a[1] != 0ull) return true;
    return a[0] >= b;
}

__device__ __forceinline__ void sub256_u64_inplace(uint64_t a[4], uint64_t dec) {
    uint64_t borrow = (a[0] < dec) ? 1ull : 0ull;
    a[0] = a[0] - dec;
#pragma unroll
    for (int i = 1; i < 4; ++i) {
        const uint64_t ai = a[i];
        a[i] = ai - borrow;
        if (!borrow) break;
        borrow = (ai < 1ull) ? 1ull : 0ull;
    }
}

__device__ __forceinline__ unsigned long long warp_reduce_add_ull(unsigned long long v) {
    unsigned mask = 0xFFFFFFFFu;
    v += __shfl_down_sync(mask, v, 16);
    v += __shfl_down_sync(mask, v, 8);
    v += __shfl_down_sync(mask, v, 4);
    v += __shfl_down_sync(mask, v, 2);
    v += __shfl_down_sync(mask, v, 1);
    return v;
}
