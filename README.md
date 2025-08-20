# ⚡CUDACyclone: GPU Satoshi Puzzle Solver

Cyclone CUDA is the GPU-powered version of the **Cyclone** project, designed to achieve extreme performance in solving Satoshi puzzles on modern NVIDIA GPUs.  
Leveraging **CUDA**, **warp-level parallelism**, and **batch EC operations**, Cyclone CUDA pushes the limits of cryptographic key search.

Secp256k1 math is based on the excellent work from [JeanLucPons/VanitySearch](https://github.com/JeanLucPons/VanitySearch), with major CUDA-specific modifications.  
Special thanks to Jean-Luc Pons for his foundational contributions to the cryptographic community.

---

## 🚀 Key Features

- **GPU Acceleration**: Optimized for NVIDIA GPUs with full CUDA support.
- **Massive Parallelism**: Tens of thousands of threads computing elliptic curve points and **hash160** simultaneously.
- **Batch EC Operations**: Efficient group addition and modular inversion with warp-level optimizations.
- **Grid/Batch Control**: Fully configurable GPU execution with `--grid` parameter (threads per batch × points per batch).
- **Cross-Platform**: Works on Linux and Windows (via WSL2 or MinGW cross-compilation).
- **Cross Architecture**: Automatic compilation for different architectures (75 86 89).
---

## 🔷 Example Output

Below is an example run of **Cyclone CUDA**:

```bash
root@ubuntu:/home/ubuntu/Work/Cyclone/CUDACyclone-work# ./CUDACyclone --range 2000000000:3FFFFFFFFF --address 1HBtApAFA9B2YZw3G2YKSMCtb3dVnjuNe2 --grid 512,256
======== PrePhase: GPU Information ====================
Device               : NVIDIA GeForce RTX 4060 (compute 8.9)
SM                   : 24
ThreadsPerBlock      : 256
Blocks               : 4096
Points batch size    : 512
Batches/SM           : 256
Memory utilization   : 6.9% (538.3 MB / 7.63 GB)
------------------------------------------------------- 
Total threads        : 1048576

======== Phase-1: Brooteforce =========================
Time: 8.0 s | Speed: 1268.9 Mkeys/s | Count: 10204470016 | Progress: 7.42 %

======== FOUND MATCH! =================================
Private Key   : 00000000000000000000000000000000000000000000000000000022382FACD0
Public Key    : 03C060E1E3771CBECCB38E119C2414702F3F5181A89652538851D2E3886BDD70C6

```

## 🛠️ Getting Started
To get started with CUDACyclone, clone the repository and type **make**

## ✌️**TIPS**
BTC: bc1qtq4y9l9ajeyxq05ynq09z8p52xdmk4hqky9c8n
