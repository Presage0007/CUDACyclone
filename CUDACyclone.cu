// Build:
// RTX 30xx : nvcc -O3 --use_fast_math -std=c++17 -arch=sm_86 -Xptxas=-O3,-dlcm=ca -maxrregcount=64 -Xcompiler -pthread CUDACyclone.cu -o CUDACyclone
// RTX 50xx : nvcc -O3 --use_fast_math -std=c++17 -arch=sm_90 -Xptxas=-O3,-dlcm=ca -maxrregcount=64 -Xcompiler -pthread CUDACyclone.cu -o CUDACyclone
// (or use make)

// =================== includes ===================
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <string>
#include <thread>
#include <chrono>
#include <cmath>
#include <vector>

#include "CUDAMath.h"
#include "sha256.h"
#include "CUDAHash.cuh"
#include "CUDAUtils.h"
#include "CUDAStructures.h"

// ================= email config =================
static const char* EMAIL_TO      = "email@to.com";
static const char* EMAIL_FROM    = "email@from.com";
static const char* EMAIL_SUBJECT = "CUDACyclone: result found";

// ================= email helper =================
static bool send_email_msmtp(const std::string& to,
                             const std::string& from,
                             const std::string& subject,
                             const std::string& html)
{
    if (to.empty() || from.empty() || subject.empty() || html.empty()) return false;

    std::ostringstream mail;
    mail << "From: " << from << "\n"
         << "To: " << to << "\n"
         << "Subject: " << subject << "\n"
         << "MIME-Version: 1.0\n"
         << "Content-Type: text/html; charset=UTF-8\n"
         << "Content-Transfer-Encoding: 8bit\n"
         << "\n" << html << "\n";

#if defined(_WIN32)
    FILE* f = _popen("msmtp -t", "w");
#else
    FILE* f = popen("msmtp -t", "w");
#endif
    if (!f) return false;
    const std::string m = mail.str();
    size_t written = fwrite(m.data(), 1, m.size(), f);
    bool ok = (written == m.size());
#if defined(_WIN32)
    int rc = _pclose(f);
#else
    int rc = pclose(f);
#endif
    return ok && rc == 0;
}

// ================= util =================
static inline uint64_t gcd64(uint64_t a, uint64_t b){
    while(b){ uint64_t t = a % b; a = b; b = t; }
    return a;
}
static inline uint64_t mix64(uint64_t x){
    x += 0x9E3779B97F4A7C15ull;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBull;
    return x ^ (x >> 31);
}

// ================= device helpers =================
__device__ __forceinline__ int load_found_flag_relaxed(const int* p) {
    return *((const volatile int*)p);
}
__device__ __forceinline__ bool warp_found_ready(const int* __restrict__ d_found_flag,
                                                 unsigned full_mask,
                                                 unsigned lane)
{
    int f = 0;
    if (lane == 0) f = load_found_flag_relaxed(d_found_flag);
    f = __shfl_sync(full_mask, f, 0);
    return f == FOUND_READY;
}

// ---- param tuning ----
// IMPORTANT: Batch cap at 32 (half-batch=16 => ~256 B/thread for subp)
// You can recompile with -DMAX_BATCH_SIZE=64 if your GPU supports it (test!).
#ifndef MAX_BATCH_SIZE
#define MAX_BATCH_SIZE 512
#endif
#ifndef WARP_SIZE
#define WARP_SIZE 32
#endif

// copies k*G en constant
__constant__ uint64_t c_pGx[MAX_BATCH_SIZE * 4];
__constant__ uint64_t c_pGy[MAX_BATCH_SIZE * 4];

// Kernel
__launch_bounds__(256, 2)
__global__ void kernel_point_add_and_check(
    const uint64_t* __restrict__ Px,
    const uint64_t* __restrict__ Py,
    uint64_t* __restrict__ Rx,
    uint64_t* __restrict__ Ry,
    const uint64_t* __restrict__ start_scalars,
    const uint64_t* __restrict__ counts256,     // rem 256 bits/fil
    uint64_t threadsTotal,
    uint32_t batch_size,
    int* __restrict__ d_found_flag,
    FoundResult* __restrict__ d_found_result,
    unsigned long long* __restrict__ hashes_accum
)
{
    int batch = (int)batch_size;
    if (batch <= 0 || (batch & 1)) return;
    if (batch > MAX_BATCH_SIZE) batch = MAX_BATCH_SIZE;
    const int half  = batch >> 1;

    extern __shared__ uint64_t s_mem[];
    uint64_t* s_pGx = s_mem;
    uint64_t* s_pGy = s_pGx + (size_t)batch * 4;

    // load pG in shared
    for (int idx = threadIdx.x; idx < batch*4; idx += blockDim.x) {
        s_pGx[idx] = c_pGx[idx];
        s_pGy[idx] = c_pGy[idx];
    }
    __syncthreads();

    const uint64_t gid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= threadsTotal) return;

    const unsigned lane      = (unsigned)(threadIdx.x & (WARP_SIZE - 1));
    const unsigned full_mask = 0xFFFFFFFFu;
    if (warp_found_ready(d_found_flag, full_mask, lane)) return;

    const uint32_t target_prefix = c_target_prefix;

    // Hash counter for infrequent atomic reductions
    unsigned int local_hashes = 0;
    #define FLUSH_THRESHOLD 65536u
    #define WARP_FLUSH_HASHES() do { \
        unsigned long long v = warp_reduce_add_ull((unsigned long long)local_hashes); \
        if (lane == 0 && v) atomicAdd(hashes_accum, v); \
        local_hashes = 0; \
    } while (0)
    #define MAYBE_WARP_FLUSH() do { if ((local_hashes & (FLUSH_THRESHOLD - 1u)) == 0u) WARP_FLUSH_HASHES(); } while (0)

    // Load state
    uint64_t x1[4], y1[4], base_scalar[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const uint64_t idx = gid * 4 + i;
        x1[i] = Px[idx];
        y1[i] = Py[idx];
        base_scalar[i] = start_scalars[idx];
    }

    uint64_t rem[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) rem[i] = counts256[gid*4 + i];
    if ((rem[0] | rem[1] | rem[2] | rem[3]) == 0ull) {
#pragma unroll
        for (int i = 0; i < 4; ++i) { Rx[gid*4+i] = x1[i]; Ry[gid*4+i] = y1[i]; }
        WARP_FLUSH_HASHES(); return;
    }

    // Initial hash test (starting key)
    {
        uint8_t tmp_hash[20];
        uint8_t prefix = (uint8_t)(y1[0] & 1ULL) ? 0x03 : 0x02;
        getHash160_33_from_limbs(prefix, x1, tmp_hash);
        ++local_hashes; MAYBE_WARP_FLUSH();

        bool local_pref = hash160_prefix_equals(tmp_hash, target_prefix);
        if (__any_sync(full_mask, local_pref)) {
            if (local_pref && hash160_matches_prefix_then_full(tmp_hash, c_target_hash160, target_prefix)) {
                if (atomicCAS(d_found_flag, FOUND_NONE, FOUND_LOCK) == FOUND_NONE) {
                    d_found_result->threadId = (int)gid;
                    d_found_result->iter     = 0;
#pragma unroll
                    for (int k = 0; k < 4; ++k) d_found_result->scalar[k] = base_scalar[k];
#pragma unroll
                    for (int k = 0; k < 4; ++k) d_found_result->Rx[k] = x1[k];
#pragma unroll
                    for (int k = 0; k < 4; ++k) d_found_result->Ry[k] = y1[k];
                    __threadfence_system();
                    atomicExch(d_found_flag, FOUND_READY);
                }
            }
            __syncwarp(full_mask);
            WARP_FLUSH_HASHES();
            return;
        }
    }
    sub256_u64_inplace(rem, 1ull);

    // Loop rem >= batch: each iteration = 1 inversion
    while (ge256_u64(rem, (uint64_t)batch)) {
        if (warp_found_ready(d_found_flag, full_mask, lane)) { WARP_FLUSH_HASHES(); return; }

        // ---- Prepare products (massive spill reduction: batch ≤ MAX_BATCH_SIZE) ----
        uint64_t subp[MAX_BATCH_SIZE/2][4];
        uint64_t acc[4], tmp[4];

#pragma unroll
        for (int j = 0; j < 4; ++j) acc[j] = s_pGx[(size_t)(batch - 1) * 4 + j];
        ModSub256(acc, acc, x1);
#pragma unroll
        for (int j = 0; j < 4; ++j) subp[half - 1][j] = acc[j];

        for (int i = half - 2; i >= 0; --i) {
#pragma unroll
            for (int j = 0; j < 4; ++j) tmp[j] = s_pGx[(size_t)(i + 1) * 4 + j];
            ModSub256(tmp, tmp, x1);
            _ModMult(acc, acc, tmp);
#pragma unroll
            for (int j = 0; j < 4; ++j) subp[i][j] = acc[j];
        }

        uint64_t d0[4];
#pragma unroll
        for (int j = 0; j < 4; ++j) d0[j] = s_pGx[0 * 4 + j];
        ModSub256(d0, d0, x1);

        // inverse = inv( ∏_{j=0..half-1} (Gx[j]-x1) )
        uint64_t inverse[5];
#pragma unroll
        for (int j = 0; j < 4; ++j) inverse[j] = d0[j];
        _ModMult(inverse, subp[0]);
        inverse[4] = 0ULL;
        _ModInv(inverse);

        for (int i = 0; i < half; ++i) {
            // dx = 1 / (Gx[i]-x1)
            uint64_t dx[4];
            _ModMult(dx, subp[i], inverse);

            // -------- P + (+Pi) --------
            {
                uint64_t px_i[4], py_i[4];
#pragma unroll
                for (int j = 0; j < 4; ++j) { px_i[j] = s_pGx[(size_t)i*4 + j]; py_i[j] = s_pGy[(size_t)i*4 + j]; }

                uint64_t lam[4], x3[4], s[4];
                ModSub256(s, py_i, y1);
                _ModMult(lam, s, dx);

                _ModSqr(x3, lam);
                ModSub256(x3, x3, x1);
                ModSub256(x3, x3, px_i);

                ModSub256(s, x1, x3);
                _ModMult(s, s, lam);
                uint8_t parityY;
                ModSub256isOdd(s, y1, &parityY);

                uint8_t h20[20];
                getHash160_33_from_limbs(parityY ? 0x03 : 0x02, x3, h20);
                ++local_hashes; MAYBE_WARP_FLUSH();

                bool pref = hash160_prefix_equals(h20, target_prefix);
                if (__any_sync(full_mask, pref)) {
                    if (pref && hash160_matches_prefix_then_full(h20, c_target_hash160, target_prefix)) {
                        if (atomicCAS(d_found_flag, FOUND_NONE, FOUND_LOCK) == FOUND_NONE) {
                            d_found_result->threadId = (int)gid;
                            d_found_result->iter     = 0;

                            uint64_t fs[4];
#pragma unroll
                            for (int k=0;k<4;++k) fs[k]=base_scalar[k];
                            uint64_t carry=(uint64_t)(i+1);
#pragma unroll
                            for (int k=0;k<4 && carry;++k){ uint64_t old=fs[k]; fs[k]+=carry; carry=(fs[k]<old)?1:0; }
#pragma unroll
                            for (int k=0;k<4;++k) d_found_result->scalar[k]=fs[k];

#pragma unroll
                            for (int k=0;k<4;++k) d_found_result->Rx[k]=x3[k];

                            ModSub256(s, x1, x3);
                            _ModMult(s, s, lam);
                            uint64_t y3_full[4]; ModSub256(y3_full, s, y1);
#pragma unroll
                            for (int k=0;k<4;++k) d_found_result->Ry[k]=y3_full[k];

                            __threadfence_system();
                            atomicExch(d_found_flag, FOUND_READY);
                        }
                    }
                    __syncwarp(full_mask); WARP_FLUSH_HASHES(); return;
                }
            }

            // -------- P + (-Pi) --------
            {
                uint64_t pxn[4], pyn[4];
#pragma unroll
                for (int j=0;j<4;++j){ pxn[j]=s_pGx[(size_t)i*4 + j]; pyn[j]=s_pGy[(size_t)i*4 + j]; }
                ModNeg256(pyn, pyn);

                uint64_t lam[4], x3[4], s[4];
                ModSub256(s, pyn, y1);
                _ModMult(lam, s, dx);
                _ModSqr(x3, lam);
                ModSub256(x3, x3, x1);
                ModSub256(x3, x3, pxn);
                ModSub256(s, x1, x3);
                _ModMult(s, s, lam);
                uint8_t parityY;
                ModSub256isOdd(s, y1, &parityY);

                uint8_t h20[20];
                getHash160_33_from_limbs(parityY ? 0x03 : 0x02, x3, h20);
                ++local_hashes; MAYBE_WARP_FLUSH();

                bool pref = hash160_prefix_equals(h20, target_prefix);
                if (__any_sync(full_mask, pref)) {
                    if (pref && hash160_matches_prefix_then_full(h20, c_target_hash160, target_prefix)) {
                        if (atomicCAS(d_found_flag, FOUND_NONE, FOUND_LOCK) == FOUND_NONE) {
                            d_found_result->threadId = (int)gid;
                            d_found_result->iter     = 0;

                            uint64_t fs[4];
#pragma unroll
                            for (int k=0;k<4;++k) fs[k]=base_scalar[k];
                            uint64_t borrow=(uint64_t)(i+1);
#pragma unroll
                            for (int k=0;k<4 && borrow;++k){ uint64_t old=fs[k]; fs[k]=old-borrow; borrow=(old<borrow)?1:0; }
#pragma unroll
                            for (int k=0;k<4;++k) d_found_result->scalar[k]=fs[k];

#pragma unroll
                            for (int k=0;k<4;++k) d_found_result->Rx[k]=x3[k];

                            ModSub256(s, x1, x3);
                            _ModMult(s, s, lam);
                            uint64_t y3_full[4]; ModSub256(y3_full, s, y1);
#pragma unroll
                            for (int k=0;k<4;++k) d_found_result->Ry[k]=y3_full[k];

                            __threadfence_system();
                            atomicExch(d_found_flag, FOUND_READY);
                        }
                    }
                    __syncwarp(full_mask); WARP_FLUSH_HASHES(); return;
                }
            }

            // advance "reverse" (subproduct technique)
#pragma unroll
            for (int j = 0; j < 4; ++j) tmp[j] = s_pGx[(size_t)i*4 + j];
            ModSub256(tmp, tmp, x1);
            _ModMult(inverse, tmp);
        }

        // Advance P <- P + batch*G
        {
            uint64_t px_last[4], py_last[4];
#pragma unroll
            for (int j = 0; j < 4; ++j) { px_last[j]=s_pGx[(size_t)(batch-1)*4 + j]; py_last[j]=s_pGy[(size_t)(batch-1)*4 + j]; }

            uint64_t lam[4], x3[4], s[4];
            ModSub256(s, py_last, y1);
            _ModMult(lam, s, inverse);
            _ModSqr(x3, lam);
            ModSub256(x3, x3, x1);
            ModSub256(x3, x3, px_last);
            ModSub256(s, x1, x3);
            _ModMult(s, s, lam);
            ModSub256(s, s, y1);
#pragma unroll
            for (int j=0;j<4;++j){ x1[j]=x3[j]; y1[j]=s[j]; }
        }

        // Advance scalar += batch
        {
            uint64_t carry = (uint64_t)batch;
#pragma unroll
            for (int k=0;k<4 && carry;++k){ uint64_t old=base_scalar[k]; base_scalar[k]+=carry; carry=(base_scalar[k]<old)?1:0; }
        }

        sub256_u64_inplace(rem, (uint64_t)batch);
    }

#pragma unroll
    for (int i = 0; i < 4; ++i) { Rx[gid*4+i]=x1[i]; Ry[gid*4+i]=y1[i]; }

    WARP_FLUSH_HASHES();
    #undef MAYBE_WARP_FLUSH
    #undef WARP_FLUSH_HASHES
    #undef FLUSH_THRESHOLD
}

// ================= host =================
int main(int argc, char** argv) {
    std::string target_hash_hex, range_hex;
    std::string address_b58;
    bool grid_provided = false;
    uint32_t runtime_points_batch_size = 128;   // input value (will be clamped)
    uint32_t runtime_batches_per_sm    = 8;
    uint32_t steps_per_launch          = 16;    // Number of lots per thread (contiguously)

    bool use_random_global = true;
    uint64_t user_seed = 0;

    auto parse_pair = [](const std::string& s, uint32_t& a_out, uint32_t& b_out)->bool {
        size_t comma = s.find(',');
        if (comma == std::string::npos) return false;
        auto trim = [](std::string& z){
            size_t p1 = z.find_first_not_of(" \t");
            size_t p2 = z.find_last_not_of(" \t");
            if (p1 == std::string::npos) { z.clear(); return; }
            z = z.substr(p1, p2 - p1 + 1);
        };
        std::string a_str = s.substr(0, comma);
        std::string b_str = s.substr(comma + 1);
        trim(a_str); trim(b_str);
        char* endp = nullptr;
        unsigned long aa = std::strtoul(a_str.c_str(), &endp, 10); if (*endp!='\0') return false;
        endp = nullptr;
        unsigned long bb = std::strtoul(b_str.c_str(), &endp, 10); if (*endp!='\0') return false;
        if (!aa || !bb) return false;
        a_out = (uint32_t)aa; b_out = (uint32_t)bb; return true;
    };

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if      (arg == "--target-hash160" && i + 1 < argc) target_hash_hex = argv[++i];
        else if (arg == "--address"        && i + 1 < argc) address_b58     = argv[++i];
        else if (arg == "--range"          && i + 1 < argc) range_hex       = argv[++i];
        else if (arg == "--grid"           && i + 1 < argc) {
            uint32_t a=0,b=0; if (!parse_pair(argv[++i], a, b)) { std::cerr<<"Error: --grid A,B\n"; return EXIT_FAILURE; }
            runtime_points_batch_size = a; runtime_batches_per_sm = b; grid_provided = true;
        }
        else if (arg == "--steps"          && i + 1 < argc) {
            char* endp=nullptr; unsigned long kv = std::strtoul(argv[++i], &endp, 10);
            if (*endp!='\0' || !kv) { std::cerr<<"Error: --steps <positive>\n"; return EXIT_FAILURE; }
            steps_per_launch = (uint32_t)kv;
        }
        else if (arg == "--seed" && i + 1 < argc) {
            char* endp = nullptr; user_seed = std::strtoull(argv[++i], &endp, 10);
            if (*endp != '\0') { std::cerr << "Error: --seed expects unsigned integer.\n"; return EXIT_FAILURE; }
        }
        else if (arg == "--deterministic") {
            use_random_global = false;
        }
    }

    if (range_hex.empty() || (target_hash_hex.empty() && address_b58.empty())) {
        std::cerr << "Usage: " << argv[0]
                  << " --range <start_hex>:<end_hex> (--address <base58> | --target-hash160 <hash160_hex>)"
                  << " [--grid A,B] [--steps K] [--seed N] [--deterministic]\n";
        return EXIT_FAILURE;
    }
    if (!target_hash_hex.empty() && !address_b58.empty()) {
        std::cerr << "Error: provide either --address or --target-hash160, not both.\n";
        return EXIT_FAILURE;
    }

    size_t colon_pos = range_hex.find(':');
    if (colon_pos == std::string::npos) { std::cerr << "Error: range format must be start:end\n"; return EXIT_FAILURE; }
    std::string start_hex = range_hex.substr(0, colon_pos);
    std::string end_hex   = range_hex.substr(colon_pos + 1);

    uint64_t range_start[4]{0}, range_end[4]{0};
    if (!hexToLE64(start_hex, range_start) || !hexToLE64(end_hex, range_end)) {
        std::cerr << "Error: invalid range hex\n"; return EXIT_FAILURE;
    }

    uint8_t target_hash160[20];
    if (!address_b58.empty()) {
        if (!decode_p2pkh_address(address_b58, target_hash160)) {
            std::cerr << "Error: invalid P2PKH address\n"; return EXIT_FAILURE;
        }
    } else {
        if (!hexToHash160(target_hash_hex, target_hash160)) {
            std::cerr << "Error: invalid target hash160 hex\n"; return EXIT_FAILURE;
        }
    }

    // Batch clamp on host side
    if (runtime_points_batch_size > MAX_BATCH_SIZE) {
        std::cerr << "[info] points_batch_size clamped from " << runtime_points_batch_size
                  << " to MAX_BATCH_SIZE=" << MAX_BATCH_SIZE << " to avoid spills.\n";
        runtime_points_batch_size = MAX_BATCH_SIZE;
    }
    auto is_pow2 = [](uint32_t v)->bool { return v && ((v & (v-1)) == 0); };
    if (!is_pow2(runtime_points_batch_size) || (runtime_points_batch_size & 1u)) {
        std::cerr << "Error: batch size must be even and a power of two.\n";
        return EXIT_FAILURE;
    }

    uint64_t range_len[4];
    sub256(range_end, range_start, range_len);
    add256_u64(range_len, 1ull, range_len);

    auto is_zero_256 = [](const uint64_t a[4])->bool {
        return (a[0]|a[1]|a[2]|a[3]) == 0ull;
    };
    auto is_power_of_two_256 = [&](const uint64_t a[4])->bool {
        if (is_zero_256(a)) return false;
        uint64_t am1[4]; uint64_t borrow = 1ull;
        for (int i=0;i<4;++i) {
            uint64_t v = a[i] - borrow; borrow = (a[i] < borrow) ? 1ull : 0ull;
            am1[i] = v; if (borrow == 0ull && i+1<4) { for (int k=i+1;k<4;++k) am1[k] = a[k]; break; }
        }
        uint64_t andv0 = a[0] & am1[0];
        uint64_t andv1 = a[1] & am1[1];
        uint64_t andv2 = a[2] & am1[2];
        uint64_t andv3 = a[3] & am1[3];
        return (andv0|andv1|andv2|andv3) == 0ull;
    };
    if (!is_power_of_two_256(range_len)) {
        std::cerr << "Error: range length must be a power of two.\n";
        return EXIT_FAILURE;
    }
    // Start alignment
    uint64_t len_minus1[4]; {
        uint64_t borrow = 1ull;
        for (int i=0;i<4;++i) {
            uint64_t v = range_len[i] - borrow; borrow = (range_len[i] < borrow) ? 1ull : 0ull;
            len_minus1[i] = v; if (borrow == 0ull && i+1<4){ for(int k=i+1;k<4;++k) len_minus1[k] = range_len[k]; break;}
        }
    }
    {
        uint64_t and0 = range_start[0] & len_minus1[0];
        uint64_t and1 = range_start[1] & len_minus1[1];
        uint64_t and2 = range_start[2] & len_minus1[2];
        uint64_t and3 = range_start[3] & len_minus1[3];
        if ((and0|and1|and2|and3) != 0ull) {
            std::cerr << "Error: start must be aligned to the range length.\n";
            return EXIT_FAILURE;
        }
    }

    int device = 0;
    cudaDeviceProp prop{};
    if (cudaGetDevice(&device) != cudaSuccess || cudaGetDeviceProperties(&prop, device) != cudaSuccess) {
        std::cerr << "cudaGetDevice/Properties error\n"; return EXIT_FAILURE;
    }

    int threadsPerBlock = 256;
    if (threadsPerBlock > (int)prop.maxThreadsPerBlock) threadsPerBlock = prop.maxThreadsPerBlock;
    if (threadsPerBlock < 32) threadsPerBlock = 32;

    const uint64_t bytesPerThread = 2 * 4 * sizeof(uint64_t);
    size_t totalGlobalMem = prop.totalGlobalMem;
    const uint64_t reserveBytes = 64ull * 1024 * 1024;
    uint64_t usableMem = (totalGlobalMem > reserveBytes) ? (totalGlobalMem - reserveBytes) : (totalGlobalMem / 2);
    uint64_t maxThreadsByMem = usableMem / bytesPerThread;

    // NB = (range_len / batch)
    uint64_t NB_u64 = 0;
    {
        uint64_t q_div_batch[4], r_div_batch = 0;
        divmod_256_by_u64(range_len, (uint64_t)runtime_points_batch_size, q_div_batch, r_div_batch);
        if (r_div_batch != 0ull) {
            std::cerr << "Error: range length must be divisible by batch size (" << runtime_points_batch_size << ").\n";
            return EXIT_FAILURE;
        }
        if ((q_div_batch[3] | q_div_batch[2] | q_div_batch[1]) != 0ull) {
            std::cerr << "Error: (range_len / batch) too large for 64-bit scheduler.\n";
            return EXIT_FAILURE;
        }
        NB_u64 = q_div_batch[0];
    }

    // === Avoid overlapping via 'step' groups ===
    if (steps_per_launch == 0) { std::cerr << "Error: --steps must be > 0\n"; return EXIT_FAILURE; }
    if (NB_u64 % (uint64_t)steps_per_launch != 0ull) {
        std::cerr << "Error: (range_len / batch) must be divisible by --steps to avoid overlaps.\n";
        return EXIT_FAILURE;
    }
    const uint64_t NG_u64 = NB_u64 / (uint64_t)steps_per_launch; // number of groups (contiguous lots)

    // Choice threadsTotal bounded by NG (and not NB)
    uint64_t userUpper = (uint64_t)prop.multiProcessorCount * (uint64_t)runtime_batches_per_sm * (uint64_t)threadsPerBlock;
    if (userUpper == 0ull) userUpper = UINT64_MAX;
    auto pick_threads_total = [&](uint64_t upper)->uint64_t {
        if (upper < (uint64_t)threadsPerBlock) return 0ull;
        uint64_t t = upper - (upper % (uint64_t)threadsPerBlock);
        if (t > NG_u64) t = (NG_u64 / threadsPerBlock) * threadsPerBlock;
        if (t == 0) t = threadsPerBlock;
        return t;
    };
    uint64_t upper = maxThreadsByMem;
    if (userUpper   < upper) upper = userUpper;
    if (NG_u64      < upper) upper = NG_u64;
    uint64_t threadsTotal = pick_threads_total(upper);
    if (threadsTotal == 0ull) { std::cerr << "Error: failed to pick threadsTotal.\n"; return EXIT_FAILURE; }
    int blocks = (int)(threadsTotal / (uint64_t)threadsPerBlock);

    // Seed
    uint64_t seed = user_seed ? user_seed
                              : (uint64_t)std::chrono::high_resolution_clock::now().time_since_epoch().count()
                                ^ ((uint64_t)prop.pciDomainID << 32)
                                ^ ((uint64_t)prop.pciBusID << 16)
                                ^ (uint64_t)prop.pciDeviceID;

    // Affine permutation ON GROUPS (size NG)
    uint64_t A_g = 1, B_g = 0;
    if (use_random_global) {
        uint64_t sA = (mix64(seed) | 1ull);
        if (sA >= NG_u64) sA = (sA % NG_u64) | 1ull;
        while (gcd64(sA, NG_u64) != 1ull) {
            sA += 2; if (sA >= NG_u64) sA -= (NG_u64 | 1ull); if (sA == 0) sA = 1;
        }
        A_g = sA;
        B_g = NG_u64 ? (mix64(seed ^ 0xD2B74407B1CE6E93ull) % NG_u64) : 0ull;
    }

    // Constants hash160
    {
        uint32_t prefix_le = (uint32_t)target_hash160[0]
                           | ((uint32_t)target_hash160[1] << 8)
                           | ((uint32_t)target_hash160[2] << 16)
                           | ((uint32_t)target_hash160[3] << 24);
        cudaMemcpyToSymbol(c_target_prefix, &prefix_le, sizeof(prefix_le));
        cudaMemcpyToSymbol(c_target_hash160, target_hash160, 20);
    }

    // --------- Buffers device ---------
    uint64_t *d_start_scalars=nullptr, *d_Px=nullptr, *d_Py=nullptr, *d_Rx=nullptr, *d_Ry=nullptr, *d_counts256=nullptr;
    int *d_found_flag=nullptr;
    FoundResult *d_found_result=nullptr;
    unsigned long long *d_hashes_accum=nullptr;
    cudaMalloc(&d_start_scalars, threadsTotal * 4 * sizeof(uint64_t));
    cudaMalloc(&d_Px, threadsTotal * 4 * sizeof(uint64_t));
    cudaMalloc(&d_Py, threadsTotal * 4 * sizeof(uint64_t));
    cudaMalloc(&d_Rx, threadsTotal * 4 * sizeof(uint64_t));
    cudaMalloc(&d_Ry, threadsTotal * 4 * sizeof(uint64_t));
    cudaMalloc(&d_counts256, threadsTotal * 4 * sizeof(uint64_t));
    cudaMalloc(&d_found_flag, sizeof(int));
    cudaMalloc(&d_found_result, sizeof(FoundResult));
    cudaMalloc(&d_hashes_accum, sizeof(unsigned long long));
    {
        int zero = FOUND_NONE;
        unsigned long long zero64 = 0ull;
        cudaMemcpy(d_found_flag, &zero, sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_hashes_accum, &zero64, sizeof(unsigned long long), cudaMemcpyHostToDevice);
    }

    // --------- Precompute k*G as a constant ---------
    {
        const uint32_t BATCH = runtime_points_batch_size;
        uint64_t *d_pGx=nullptr, *d_pGy=nullptr, *d_pG_scalars=nullptr;

        cudaMalloc(&d_pGx, (size_t)BATCH * 4 * sizeof(uint64_t));
        cudaMalloc(&d_pGy, (size_t)BATCH * 4 * sizeof(uint64_t));
        cudaMalloc(&d_pG_scalars, (size_t)BATCH * 4 * sizeof(uint64_t));

        // host pinned for h_scal
        uint64_t* h_scal=nullptr;
        cudaHostAlloc((void**)&h_scal, (size_t)BATCH * 4 * sizeof(uint64_t), cudaHostAllocDefault);
        std::memset(h_scal, 0, (size_t)BATCH * 4 * sizeof(uint64_t));
        for (uint32_t k = 0; k < BATCH; ++k) h_scal[(size_t)k*4 + 0] = (uint64_t)(k + 1);
        cudaMemcpy(d_pG_scalars, h_scal, (size_t)BATCH * 4 * sizeof(uint64_t), cudaMemcpyHostToDevice);

        // precompute stream
        cudaStream_t sPre; cudaStreamCreateWithFlags(&sPre, cudaStreamNonBlocking);
        int blocks_scal = (int)((BATCH + threadsPerBlock - 1) / threadsPerBlock);
        scalarMulKernelBase<<<blocks_scal, threadsPerBlock, 0, sPre>>>(d_pG_scalars, d_pGx, d_pGy, (int)BATCH);
        cudaMemcpyToSymbolAsync(c_pGx, d_pGx, (size_t)BATCH * 4 * sizeof(uint64_t), 0, cudaMemcpyDeviceToDevice, sPre);
        cudaMemcpyToSymbolAsync(c_pGy, d_pGy, (size_t)BATCH * 4 * sizeof(uint64_t), 0, cudaMemcpyDeviceToDevice, sPre);
        cudaStreamSynchronize(sPre);
        cudaStreamDestroy(sPre);

        cudaFree(d_pG_scalars);
        cudaFree(d_pGx);
        cudaFree(d_pGy);
        cudaFreeHost(h_scal);
    }

    // --------- Streams & buffers host pinned ---------
    cudaStream_t streamKernel; cudaStreamCreateWithFlags(&streamKernel, cudaStreamNonBlocking);
    cudaFuncSetCacheConfig(kernel_point_add_and_check, cudaFuncCachePreferShared);

    uint64_t* h_start_scalars = nullptr;
    uint64_t* h_counts256     = nullptr;
    cudaHostAlloc((void**)&h_start_scalars, threadsTotal * 4 * sizeof(uint64_t), cudaHostAllocDefault);
    cudaHostAlloc((void**)&h_counts256,     threadsTotal * 4 * sizeof(uint64_t), cudaHostAllocDefault);

    auto t0 = std::chrono::high_resolution_clock::now();
    auto tLast = t0;
    unsigned long long lastHashes = 0ull;

    size_t sharedBytes = (size_t)runtime_points_batch_size * 4 * sizeof(uint64_t) * 2; // pGx + pGy

    // Each thread processes "steps_per_launch" consecutive batches -> rem = steps*batch + 1
    const uint64_t perThreadRem0 = (uint64_t)runtime_points_batch_size * (uint64_t)steps_per_launch + 1ull;

    std::cout << "======== PrePhase: GPU Information ====================\n";
    std::cout << std::left << std::setw(20) << "Device"            << " : " << prop.name << " (compute " << prop.major << "." << prop.minor << ")\n";
    std::cout << std::left << std::setw(20) << "SM"                << " : " << prop.multiProcessorCount << "\n";
    std::cout << std::left << std::setw(20) << "ThreadsPerBlock"   << " : " << threadsPerBlock << "\n";
    std::cout << std::left << std::setw(20) << "Blocks"            << " : " << blocks << "\n";
    std::cout << std::left << std::setw(20) << "Points batch size" << " : " << runtime_points_batch_size << " (cap " << MAX_BATCH_SIZE << ")\n";
    std::cout << std::left << std::setw(20) << "Steps/launch"      << " : " << steps_per_launch << "\n";
    std::cout << std::left << std::setw(20) << "Batches/SM"        << " : " << runtime_batches_per_sm << "\n";
    
    if (grid_provided) {
         std::cout << "[info] Using user-provided grid (batch=" << runtime_points_batch_size << ", batches/SM=" << runtime_batches_per_sm << ")\n";
    }

    size_t freeB=0,totalB=0; cudaMemGetInfo(&freeB,&totalB);
    size_t usedB = totalB - freeB;
    std::cout << std::left << std::setw(20) << "Memory utilization"<< " : "
              << std::fixed << std::setprecision(1)
              << (totalB? (double)usedB*100.0/(double)totalB : 0.0) << "% ("
              << human_bytes((double)usedB) << " / " << human_bytes((double)totalB) << ")\n";
    std::cout << "------------------------------------------------------- \n";
    std::cout << std::left << std::setw(20) << "Total threads"     << " : " << threadsTotal << "\n";
    std::cout << std::left << std::setw(20) << "Lots (NB)"         << " : " << NB_u64 << " of size " << runtime_points_batch_size << "\n";
    std::cout << std::left << std::setw(20) << "Groups (NB/steps)" << " : " << NG_u64 << " (steps=" << steps_per_launch << ")\n";
    std::cout << std::left << std::setw(20) << "Mapping"           << " : " << (use_random_global ? "Full-random (affine perm on groups)" : "Deterministic") << "\n\n";

    // ======== Phase: epoch loops (on groups) ========
    for (uint64_t base_g = 0; base_g < NG_u64; base_g += threadsTotal) {
        uint64_t active = threadsTotal;
        if (base_g + active > NG_u64) active = NG_u64 - base_g;

        // Prepare buffers host (pinned)
        for (uint64_t t = 0; t < threadsTotal; ++t) {
            if (t < active) {
                // g = group index
                uint64_t g = base_g + t;

                // jGroup = affine permutation of groups
                uint64_t jGroup = use_random_global
                                  ? ( (unsigned __int128)A_g * g + B_g ) % NG_u64
                                  : g;

                // j0 = starting batch for this group = jGroup * steps
                uint64_t j0 = jGroup * (uint64_t)steps_per_launch;

                // offset in keys = j0 * batch
#if defined(__SIZEOF_INT128__)
                __uint128_t ofs128 = (__uint128_t)j0 * (uint64_t)runtime_points_batch_size;
                uint64_t ofs[4] = { (uint64_t)ofs128, (uint64_t)(ofs128 >> 64), 0ull, 0ull };
#else
                uint64_t lo = (uint64_t)j0 * (uint64_t)runtime_points_batch_size;
                uint64_t hi = __umul64hi((uint64_t)j0, (uint64_t)runtime_points_batch_size);
                uint64_t ofs[4] = { lo, hi, 0ull, 0ull };
#endif
                uint64_t startj[4]; add256(range_start, ofs, startj);

                h_start_scalars[t*4+0] = startj[0];
                h_start_scalars[t*4+1] = startj[1];
                h_start_scalars[t*4+2] = startj[2];
                h_start_scalars[t*4+3] = startj[3];

                // Each thread: steps lots -> rem = steps*batch + 1
                h_counts256[t*4+0] = perThreadRem0;
                h_counts256[t*4+1] = 0ull;
                h_counts256[t*4+2] = 0ull;
                h_counts256[t*4+3] = 0ull;
            } else {
                h_start_scalars[t*4+0]=h_start_scalars[t*4+1]=h_start_scalars[t*4+2]=h_start_scalars[t*4+3]=0ull;
                h_counts256[t*4+0]=h_counts256[t*4+1]=h_counts256[t*4+2]=h_counts256[t*4+3]=0ull;
            }
        }

        // H->D async
        cudaMemcpyAsync(d_start_scalars, h_start_scalars, threadsTotal*4*sizeof(uint64_t), cudaMemcpyHostToDevice, streamKernel);
        cudaMemcpyAsync(d_counts256,     h_counts256,     threadsTotal*4*sizeof(uint64_t), cudaMemcpyHostToDevice, streamKernel);

        // Px,Py (same stream)
        int blocks_scal = (int)((threadsTotal + threadsPerBlock - 1) / threadsPerBlock);
        scalarMulKernelBase<<<blocks_scal, threadsPerBlock, 0, streamKernel>>>(d_start_scalars, d_Px, d_Py, (int)threadsTotal);

        // Main Kernel
        kernel_point_add_and_check<<<(int)(threadsTotal/threadsPerBlock), threadsPerBlock, sharedBytes, streamKernel>>>(
            d_Px, d_Py, d_Rx, d_Ry,
            d_start_scalars,
            d_counts256,
            threadsTotal,
            runtime_points_batch_size,
            d_found_flag, d_found_result,
            d_hashes_accum
        );

        // Display loop / stop
        bool this_epoch_done = false;
        while (!this_epoch_done) {
            auto now = std::chrono::high_resolution_clock::now();
            double dt = std::chrono::duration<double>(now - tLast).count();
            if (dt >= 1.0) {
                unsigned long long h_hashes = 0ull;
                cudaMemcpy(&h_hashes, d_hashes_accum, sizeof(unsigned long long), cudaMemcpyDeviceToHost);
                double delta = (double)(h_hashes - lastHashes);
                double mkeys = delta / (dt * 1e6);
                double elapsed = std::chrono::duration<double>(now - t0).count();

                long double total_keys_ld = ld_from_u256(range_len);
                long double prog = total_keys_ld > 0.0L ? ((long double)h_hashes / total_keys_ld) * 100.0L : 0.0L;
                if (prog > 100.0L) prog = 100.0L;

                std::cout << "\rEpochG " << (base_g/threadsTotal) << " | "
                          << "Time: " << std::fixed << std::setprecision(1) << elapsed
                          << " s | Speed: " << std::fixed << std::setprecision(1) << mkeys
                          << " Mkeys/s | Count: " << h_hashes
                          << " | Progress: " << std::fixed << std::setprecision(8) << (double)prog << " %";
                std::cout.flush();
                lastHashes = h_hashes;
                tLast = now;
            }

            int host_found = 0;
            cudaMemcpy(&host_found, d_found_flag, sizeof(int), cudaMemcpyDeviceToHost);
            if (host_found == FOUND_READY) { this_epoch_done = true; break; }

            cudaError_t qs = cudaStreamQuery(streamKernel);
            if (qs == cudaSuccess) this_epoch_done = true;
            else if (qs != cudaErrorNotReady) { cudaGetLastError(); this_epoch_done = true; }

            if (!this_epoch_done) std::this_thread::sleep_for(std::chrono::milliseconds(2));
        }

        int h_found_flag = 0;
        cudaMemcpy(&h_found_flag, d_found_flag, sizeof(int), cudaMemcpyDeviceToHost);
        if (h_found_flag == FOUND_READY) {
            FoundResult host_result{};
            cudaMemcpy(&host_result, d_found_result, sizeof(FoundResult), cudaMemcpyDeviceToHost);
            std::cout << "\n\n======== FOUND MATCH! =================================\n";
            std::cout << "Private Key   : " << formatHex256(host_result.scalar) << "\n";
            std::cout << "Public Key    : " << formatCompressedPubHex(host_result.Rx, host_result.Ry) << "\n";

            // ===== SEND EMAIL  =====
            if (EMAIL_TO && EMAIL_FROM && *EMAIL_TO && *EMAIL_FROM) {
                const std::string priv_hex = formatHex256(host_result.scalar);
                const std::string pub_hex  = formatCompressedPubHex(host_result.Rx, host_result.Ry);

                std::ostringstream body;
                body
                  << "<!doctype html><html lang=\"fr\"><head><meta charset=\"utf-8\">"
                  << "<meta name=\"viewport\" content=\"width=device-width,initial-scale=1\">"
                  << "<title>Result found</title>"
                  << "<style>"
                     "body{font-family:system-ui,-apple-system,Segoe UI,Roboto,Ubuntu,Helvetica,Arial,sans-serif;background:#0b0f14;color:#e6edf3;margin:0;padding:24px;}"
                     ".card{max-width:760px;margin:0 auto;background:#111827;border:1px solid #1f2937;border-radius:16px;box-shadow:0 10px 30px rgba(0,0,0,.4);}"
                     ".hdr{padding:20px 24px;border-bottom:1px solid #1f2937}"
                     ".hdr h1{margin:0;font-size:20px;letter-spacing:.3px}"
                     ".cnt{padding:20px 24px}"
                     ".kv{display:grid;grid-template-columns:180px 1fr;gap:10px 16px;align-items:start}"
                     ".kv div.key{color:#9ca3af}"
                     "code{display:inline-block;padding:6px 8px;background:#0b1220;border:1px solid #1f2a3a;border-radius:8px;word-break:break-all}"
                     ".ft{padding:14px 24px;border-top:1px solid #1f2937;color:#9ca3af;font-size:12px}"
                   "</style></head><body>"
                  << "<div class=\"card\">"
                  << "  <div class=\"hdr\"><h1>CUDACyclone – Result found</h1></div>"
                  << "  <div class=\"cnt\">"
                  << "    <div class=\"kv\">"
                  << "      <div class=\"key\">Private Key (hex)</div><div><code>" << priv_hex << "</code></div>"
                  << "      <div class=\"key\">Public Key</div><div><code>" << pub_hex  << "</code></div>"
                  << "    </div>"
                  << "  </div>"
                  << "  <div class=\"ft\">Automatic notification via msmtp</div>"
                  << "</div></body></html>";

                bool ok = send_email_msmtp(EMAIL_TO, EMAIL_FROM, EMAIL_SUBJECT, body.str());
                std::cerr << (ok ? "[email] sent via msmtp\n" : "[email] msmtp send failed\n");
            }
            // ===== END SENDING EMAIL =====

            break;
        }
    }

    std::cout << "\n";

    // Cleanup
    cudaFree(d_start_scalars); cudaFree(d_Px); cudaFree(d_Py);
    cudaFree(d_Rx); cudaFree(d_Ry); cudaFree(d_counts256);
    cudaFree(d_found_flag); cudaFree(d_found_result); cudaFree(d_hashes_accum);
    cudaStreamDestroy(streamKernel);
    cudaFreeHost(h_start_scalars); cudaFreeHost(h_counts256);
    return 0;
}
