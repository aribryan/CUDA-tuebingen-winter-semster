################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CU_SRCS += \
../testPrefix.cu 

CPP_SRCS += \
../PPM.cpp 

OBJS += \
./PPM.o \
./testPrefix.o 

CU_DEPS += \
./testPrefix.d 

CPP_DEPS += \
./PPM.d 


# Each subdirectory must supply rules for building sources it contributes
%.o: ../%.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: NVCC Compiler'
	/graphics/opt/opt_Ubuntu16.04/cuda/toolkit_9.0/cuda/bin/nvcc -O3 -gencode arch=compute_61,code=sm_61  -odir "." -M -o "$(@:%.o=%.d)" "$<"
	/graphics/opt/opt_Ubuntu16.04/cuda/toolkit_9.0/cuda/bin/nvcc -O3 --compile  -x c++ -o  "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '

%.o: ../%.cu
	@echo 'Building file: $<'
	@echo 'Invoking: NVCC Compiler'
	/graphics/opt/opt_Ubuntu16.04/cuda/toolkit_9.0/cuda/bin/nvcc -O3 -gencode arch=compute_61,code=sm_61  -odir "." -M -o "$(@:%.o=%.d)" "$<"
	/graphics/opt/opt_Ubuntu16.04/cuda/toolkit_9.0/cuda/bin/nvcc -O3 --compile --relocatable-device-code=false -gencode arch=compute_61,code=compute_61 -gencode arch=compute_61,code=sm_61  -x cu -o  "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


