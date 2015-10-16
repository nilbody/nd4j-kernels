# Every subdirectory with source files must be described here
SUBDIRS := \
src \


export TOP_DIR:= .
export OUTPUT_DIR:= $(TOP_DIR)/output
export SUB_MODULES:= pairwise_transforms reduce reduce3 scalar transforms 

all: $(SUB_MODULES)
$(SUB_MODULES):
	echo "Going in $@"
	cd src/$@ && $(MAKE) && cd ../..
default: all

clean: 
	$(shell rm -rf output/*)

nd4j-kernels: $(OBJS) $(USER_OBJS)
	@echo 'Building target: $@'
	@echo 'Invoking: NVCC Linker'
	/usr/local/cuda-7.5/bin/nvcc --cudart static --relocatable-device-code=true -gencode arch=compute_50,code=compute_50 -gencode arch=compute_50,code=sm_50 -link -o  "nd4j-kernels" $(OBJS) $(USER_OBJS) $(LIBS)
	@echo 'Finished building target: $@'
	@echo ' '
