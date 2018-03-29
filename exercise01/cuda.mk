# Uni Tübingen Computer Graphics setup
CUDA        = ${CUDA_INSTALL_PATH}/

PATH       += ${CUDA}/open64/bin:${CUDA}/bin
ARCH        = $(shell uname -m)
ifeq (${ARCH},x86_64)
  CUDALIB   = lib64
else
  CUDALIB   = lib
endif

# Variables
CUCC        = nvcc
CUPPFLAGS   = -I${CUDA}/include -I${CUDA}/include/crt
CUFLAGS     =
CULOADLIBES = -L${CUDA}/${CUDALIB}
CULDLIBS    = -lcudart
CULDFLAGS   =
TARGET_ARCH = -arch sm_61

# Macros for compiling and linking
COMPILE.cu = $(CUCC) $(CUPPFLAGS) $(CUFLAGS) $(TARGET_ARCH)
LINK.cu    = $(CUCC) $(CUPPFLAGS) $(CUFLAGS) $(CULDFLAGS) $(TARGET_ARCH)

# CUDA rules
%.cubin: %.cu
	$(COMPILE.cu) --cubin $<

%.o: %.cu
	$(COMPILE.cu) -c $<

%: %.cu
	$(LINK.cu) $^ $(CULOADLIBES) $(CULDLIBS) -o $@
