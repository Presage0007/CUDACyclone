# ⚡CUDACyclone Full random mode + deterministic: GPU Satoshi Puzzle Solver

Cyclone CUDA is the GPU-powered version of the **Cyclone** project, designed to achieve extreme performance in solving Satoshi puzzles on modern NVIDIA GPUs.  
Leveraging **CUDA**, **warp-level parallelism**, and **batch EC operations**, Cyclone CUDA pushes the limits of cryptographic key search.

Secp256k1 math is based on the excellent work from [JeanLucPons/VanitySearch](https://github.com/JeanLucPons/VanitySearch) and [FixedPaul/VanitySearch-Bitcrack](https://github.com/FixedPaul), with major CUDA-specific modifications.  
Special thanks to Jean-Luc Pons for his foundational contributions to the cryptographic community.

Cyclone CUDA is also one of the **simplest CUDA-based tools** for solving Satoshi puzzles with a GPU.  
It was designed with clarity and minimalism in mind — making it easy to **compile, understand, and run**, even for those new to CUDA programming.  

Despite its simplicity, CUDACyclone leverages **massive GPU parallelism** to achieve extreme performance in elliptic curve computations and HASH160 pipelines. 

⚠️ **Achieved over 6Gkeys/s on RTX 4090**.

---

## 🚀 Key Features

- **GPU Acceleration**: Full CUDA optimization for NVIDIA GPUs.
- **Massive Parallelism**: Tens of thousands of threads scanning keys in parallel.
- **Batch EC Operations**: One modular inversion per batch (optimized group processing).
- **Affine Permutation**: Default **full-random group permutation**, or deterministic mode.
- **Steps Per Thread**: Each thread can scan several lots per batch (`--steps`).
- **Grid Tuning**: Control over batch size and batches per SM via `--grid`.
- **Email Alert**: Sends automatic email via `msmtp` if a key is found.
- **Low VRAM Usage**: Can run on almost any CUDA-enabled GPU.
- **Cross Platform**: Runs on Linux, WSL2, Windows (cross-compiled).
- **Clean Code**: Easy to hack and extend.

---

## ⚙️ Options

- `--range <start_hex>:<end_hex>`  
  Range to search (inclusive).  
  ⚠️ **Length must be a power of 2**, and **start must be aligned** accordingly.

- `--address <base58>`  
  Target P2PKH address (e.g. 1PWo3Je...).

- `--target-hash160 <hex>`  
  Optional alternative to `--address`. Directly specify HASH160.

- `--grid A,B`  
  GPU launch configuration.  
  - `A` = Points batch size (per thread, per batch)  
  - `B` = Batches per SM  
  Example: `--grid 512,256`

- `--steps <K>`  
  Number of **consecutive lots** per thread launch.  
  Must divide the total number of lots (`range_len / batch`).  
  Default: `16`

- `--seed <N>`  
  Sets seed for random permutation. Only used in full-random mode.

- `--deterministic`  
  Disables random permutation. Scans the range sequentially, group by group.

---

## 🔀 Random vs Deterministic Mode

Let’s say the total number of **lots** = `NB = range_len / batch`, and `--steps K` means each thread scans `K` consecutive lots.  
Then the total number of **groups** = `NG = NB / K`.

- In **default (random)** mode:  
  The program visits groups in pseudo-random order using an **affine permutation**:  
  `g' = (a * g + b) mod NG`  
  with `gcd(a, NG) = 1`.  
  This ensures **100% full coverage** of the range with **no overlap** and **no repetition**, in a shuffled order.

- With `--deterministic`:  
  Groups are visited in natural order: `0, 1, 2, ... NG-1`.

---

## 📨 Email Notification (Optional)

CUDACyclone can send an email when a key is found (via `msmtp`).  
Set up `.msmtprc`:

```bash
apt install msmtp
nano ~/.msmtprc
```

Example config:
```
defaults
auth           on
tls            on
logfile        ~/.msmtp.log

account        default
host           smtp.yourprovider.com
port           587
from           your@email.com
user           your@email.com
password       yourpassword
tls_trust_file /etc/ssl/certs/ca-certificates.crt
```

Make it private:
```bash
chmod 600 ~/.msmtprc
```

In `CUDACyclone.cu`, edit:

```cpp
static const char* EMAIL_TO      = "your@email.com";
static const char* EMAIL_FROM    = "your@email.com";
static const char* EMAIL_SUBJECT = "CUDACyclone: Result Found!";
```

---

## 🔷 Community Benchmarks

| GPU               | Grid      | Speed (Mkeys/s) | Notes            |
|-------------------|-----------|-----------------|------------------|
| RTX 4090          | 128,1024  | 6214 Mkeys/s    | Community report |
| RTX 4090          | 512,512   | 6038 Mkeys/s    | Community report |
| RTX 4060          | 512,512   | 1238 Mkeys/s    | Own result       |
| RTX 4070 Ti Super | 512,1024  | 3170 Mkeys/s    | Community report |
| L4-2Q             | 512,256   | 1360 Mkeys/s    | Community report |
| RTX 3070 Mobile   | 256,256   | 1150 Mkeys/s    | Community report |

---

## 📟 Example Output

**RTX 4060**
```bash
./CUDACyclone --range 2000000000:3FFFFFFFFF --address 1HBtApAFA9B2YZw3G2YKSMCtb3dVnjuNe2 --grid 512,256
======== PrePhase: GPU Information ====================
Device               : NVIDIA GeForce RTX 4060 (compute 8.9)
SM                   : 24
ThreadsPerBlock      : 256
Blocks               : 4096
Points batch size    : 512
Steps/launch         : 16
Batches/SM           : 256
Memory utilization   : 6.9% (538.3 MB / 7.63 GB)
-------------------------------------------------------
Total threads        : 1048576
Lots (NB)            : 8388608
Groups (NB/steps)    : 524288
Mapping              : Full-random (affine permutation)

======== Phase-1: Bruteforce ==========================
Time: 8.0 s | Speed: 1268.9 Mkeys/s | Count: 10204470016 | Progress: 7.42 %

======== FOUND MATCH! =================================
Private Key   : 00000000000000000000000000000000000000000000000000000022382FACD0
Public Key    : 03C060E1E3771CBECCB38E119C2414702F3F5181A89652538851D2E3886BDD70C6
[email] sent via msmtp
```

**RTX 4090**
```bash
./CUDACyclone --range 400000000000000000:7fffffffffffffffff --target-hash160 abcd1234... --grid 512,512 --steps 32 --deterministic
======== PrePhase: GPU Information ====================
Device               : NVIDIA GeForce RTX 4090 (compute 8.9)
SM                   : 128
ThreadsPerBlock      : 256
Blocks               : 65536
Points batch size    : 512
Steps/launch         : 32
Batches/SM           : 512
Memory utilization   : 14.5% (3.42 GB / 23.5 GB)
-------------------------------------------------------
Total threads        : 16777216
Lots (NB)            : 268435456
Groups (NB/steps)    : 8388608
Mapping              : Deterministic
```

---

## 🛠️ Getting Started

Install everything on a clean Linux machine:

```bash
apt update
apt install -y joe zip screen curl libcurl4
apt install -y build-essential gcc make
apt install -y cuda-toolkit
git clone https://github.com/Dookoo2/CUDACyclone.git
cd CUDACyclone
make
```

Build for RTX 30xx:
```bash
nvcc -O3 --use_fast_math -std=c++17 -arch=sm_86 \
  -Xptxas=-O3,-dlcm=ca -maxrregcount=64 \
  -Xcompiler -pthread \
  CUDACyclone.cu -o CUDACyclone
```

Build for RTX 50xx:
```bash
nvcc -O3 --use_fast_math -std=c++17 -arch=sm_90 \
  -Xptxas=-O3,-dlcm=ca -maxrregcount=64 \
  -Xcompiler -pthread \
  CUDACyclone.cu -o CUDACyclone
```

---

## 🧪 Version History

- **v1.2** – Affine permutation on lots groups (default mode), `--steps`, `--seed`, msmtp email alert.
- **v1.1** – Constant memory for pGx/pGy to reduce thermal throttling.
- **v1.0** – Initial release.

---

## ✌️ Donate

**BTC**: `1Bitcoinfr7ZVB1cFVgLJJcweVt8JR2Z3o`

