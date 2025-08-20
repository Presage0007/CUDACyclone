#ifndef CUDA_HASH_CUH
#define CUDA_HASH_CUH

#include <cstdint>
#include <cuda_runtime.h>
#include <cstring>

struct MatchResult {
    int found;           
    uint8_t publicKey[33];
    uint8_t sha256[32];
    uint8_t ripemd160[20];
};

__device__ void getSHA256_33bytes(const uint8_t* pubkey33, uint8_t sha[32]);
__device__ void getRIPEMD160_32bytes(const uint8_t* sha, uint8_t ripemd[20]);
__device__ void getHash160_33bytes(const uint8_t* pubkey33, uint8_t* hash20);
__device__ void addBigEndian32(uint8_t* data32, uint64_t offset);
__device__ void RIPEMD160_from_SHA256_state(const uint32_t sha_state_be[8], uint8_t ripemd20[20]);
__device__ void getHash160_33_from_limbs(uint8_t prefix02_03, const uint64_t x_be_limbs[4], uint8_t out20[20]);
#endif 
