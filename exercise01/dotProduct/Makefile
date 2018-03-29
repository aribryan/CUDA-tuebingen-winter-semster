
# Add source files here
TARGET		:= testDotProduct
# Cuda source files (compiled with cudacc)
CUFILES		:= testDotProduct.cu
# C/C++ source files (compiled with gcc / c++)
#CCFILES		:= PPM.cpp


# Warning flags
CXXWARN_FLAGS := \
	-W -Wall \
	-Wswitch \
	-Wformat \
	-Wchar-subscripts \
	-Wparentheses \
	-Wmultichar \
	-Wtrigraphs \
	-Wpointer-arith \
	-Wcast-align \
	-Wreturn-type \
	-Wno-unused-function \
	$(SPACE)

CWARN_FLAGS := $(CXXWARN_FLAGS) \
	-Wstrict-prototypes \
	-Wmissing-prototypes \
	-Wmissing-declarations \
	-Wnested-externs \
	-Wmain \

CXXFLAGS += $(CXXWARN_FLAGS)

CFLAGS += $(CWARN_FLAGS)

NVCCFLAGS += -arch sm_30


################################################################################
# Rules and targets

.SUFFIXES : .cu 

# detect OS
OSUPPER = $(shell uname -s 2>/dev/null | tr [:lower:] [:upper:])
OSLOWER = $(shell uname -s 2>/dev/null | tr [:upper:] [:lower:])

# Compilers
NVCC       := $(CUDA_INSTALL_PATH)/bin/nvcc 
CXX        := g++
CC         := gcc
LINK       := g++ -fPIC

# Includes
INCLUDES  += -I$(CUDA_INSTALL_PATH)/include -I$(CUDA_SDK_DIR)/C/common/inc
CXXFLAGS += $(INCLUDES)
CFLAGS += $(INCLUDES)
NVCCFLAGS += $(INCLUDES)


ifeq ($(dbg),1)
	COMMONFLAGS += -g
	NVCCFLAGS   += -D_DEBUG -G -g -Xcompiler "-gdwarf-3"
else
	COMMONFLAGS += -O3
	NVCCFLAGS   += --compiler-options -fno-strict-aliasing
endif

CXXFLAGS += $(COMMONFLAGS)
CFLAGS += $(COMMONFLAGS)
NVCCFLAGS += $(COMMONFLAGS)

LDFLAGS += -L$(CUDA_INSTALL_PATH)/lib64 -lcudart


OBJS := $(patsubst %.cpp,%.o,$(CCFILES))
OBJS += $(patsubst %.c,%.o,$(CFILES))
OBJS += $(patsubst %.cu,%.o,$(CUFILES))

all : $(TARGET)

%.o : %.cu
	$(NVCC) $(NVCCFLAGS) -o $@ -c $<
#	$(NVCC) $(NVCCFLAGS) -o $@.ptx -ptx $<

$(TARGET) : $(OBJS)
	$(LINK) -o $(TARGET) $(OBJS) $(LDFLAGS)

clean :
	rm -f $(OBJS)
	rm -f $(TARGET)
