# ===== Settings =====
TARGET_BASE := CUDACyclone
SRC         := CUDACyclone.cu CUDAHash.cu
NVCC        ?= nvcc

# --- OS detection ---
# On GNU Make for Windows, $(OS) is usually Windows_NT; otherwise try uname.
ifeq ($(OS),Windows_NT)
  EXEEXT := .exe
  OBJEXT := obj
  RM     := del /f /q
  # Avoid shell-specific $(shell command -v ...) on Windows
  DETECTED_CC :=
  # Static cudart on Windows est pénible (libs système en plus). Reste en shared.
  CUDART  := -cudart=shared
  SHELL   := cmd
else
  EXEEXT := 
  OBJEXT := o
  RM     := rm -f
  DETECTED_CC := $(shell command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -n1 | tr -d '.' || echo)
  CUDART  := -cudart=static
  SHELL   := /bin/sh
endif

TARGET      := $(TARGET_BASE)$(EXEEXT)
OBJ         := $(patsubst %.cu,%.$(OBJEXT),$(SRC))

# --- Architectures ---
SM_ARCHS    := 75 86 89
ifneq ($(DETECTED_CC),)
  SM_ARCHS  += $(DETECTED_CC)
endif
GENCODE     := $(foreach arch,$(sort $(SM_ARCHS)),-gencode arch=compute_$(arch),code=sm_$(arch))

# --- Flags ---
NVCCFLAGS   := -O3 -rdc=true -use_fast_math -Xptxas=-O3,-dlcm=ca -Wno-deprecated-gpu-targets $(GENCODE)
CXXFLAGS    := -std=c++17
LDFLAGS     := -lcudadevrt $(CUDART)

# Pour MSVC host (Windows), on peut ajouter /bigobj /EHsc si besoin :
ifeq ($(OS),Windows_NT)
  CXXFLAGS += -Xcompiler="/EHsc /bigobj"
endif

.PHONY: all clean print-archs

all: $(TARGET)

$(TARGET): $(OBJ)
	$(NVCC) $(NVCCFLAGS) $(CXXFLAGS) $^ -o $@ $(LDFLAGS)

%.$(OBJEXT): %.cu
	$(NVCC) $(NVCCFLAGS) $(CXXFLAGS) -c $< -o $@

clean:
	-$(RM) $(TARGET) $(OBJ)

print-archs:
	@echo Architectures: $(sort $(SM_ARCHS))
	@echo GENCODE: $(GENCODE)
