# ===== Settings =====
TARGET      := CUDACyclone
SRC         := CUDACyclone.cu CUDAHash.cu
OBJ         := $(SRC:.cu=.o)
NVCC        ?= nvcc

# Optional: detect compute cap if nvidia-smi exists (WSL may not have it)
DETECTED_CC := $(shell command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -n1 | tr -d '.' || echo)

# Default architectures (add/remove as needed)
SM_ARCHS    := 75 86 89 90
ifneq ($(DETECTED_CC),)
  SM_ARCHS  += $(DETECTED_CC)
endif
# If you really need Volta:
# SM_ARCHS  += 70

GENCODE     := $(foreach arch,$(sort $(SM_ARCHS)),-gencode arch=compute_$(arch),code=sm_$(arch))

NVCCFLAGS   := -O3 -rdc=true -use_fast_math -Xptxas=-O3,-dlcm=ca -Wno-deprecated-gpu-targets $(GENCODE)
CXXFLAGS    := -std=c++17
LDFLAGS     := -lcudadevrt -cudart=static

.PHONY: all clean print-archs

all: $(TARGET)

$(TARGET): $(OBJ)
        $(NVCC) $(NVCCFLAGS) $(CXXFLAGS) $^ -o $@ $(LDFLAGS)

%.o: %.cu
        $(NVCC) $(NVCCFLAGS) $(CXXFLAGS) -c $< -o $@

clean:
        rm -f $(TARGET) $(OBJ)

print-archs:
        @echo "Architectures: $(sort $(SM_ARCHS))"
        @echo "GENCODE: $(GENCODE)"
