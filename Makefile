############################################################################################################
VERSION  = V1_$(shell date "+%d_%m_%Y_%T")
VERSION  = V1_$(shell date "+%Y_%m_%d_%H-%M-%S")
STANDARD = c99
############################################################################################################
USE_FLTK = yes
DEBUG = no
GCC ?= g++ 

#SET CUDA PATH
CUDA_PATH ?= /usr/local/cuda
#-11.0
GPU_ARCH = sm_75
GENCODE_FLAGS = -arch=$(GPU_ARCH)
#GENCODE_FLAGS = -arch=$(GPU_ARCH) --ptxas-options=-v
############################################################################################################
############################################################################################################
############################################################################################################
# OS Name (Linux or Darwin)
OSUPPER = $(shell uname -s 2>/dev/null | tr [:lower:] [:upper:])
OSLOWER = $(shell uname -s 2>/dev/null | tr [:upper:] [:lower:])
# Flags to detect 32-bit or 64-bit OS platform
OS_SIZE = $(shell uname -m | sed -e "s/i.86/32/" -e "s/x86_64/64/")
OS_ARCH = $(shell uname -m | sed -e "s/i386/i686/")
# These flags will override any settings
ifeq ($(i386),1)
	OS_SIZE = 32
	OS_ARCH = i686
endif
ifeq ($(x86_64),1)
	OS_SIZE = 64
	OS_ARCH = x86_64
endif
# Flags to detect either a Linux system (linux) or Mac OSX (darwin)
DARWIN = $(strip $(findstring DARWIN, $(OSUPPER)))
# Location of the CUDA Toolkit binaries and libraries
#CHANGE TO YOUR CUDA PATH
CUDA_INC_PATH   ?= $(CUDA_PATH)/include
CUDA_BIN_PATH   ?= $(CUDA_PATH)/bin
ifneq ($(DARWIN),)
  CUDA_LIB_PATH  ?= $(CUDA_PATH)/lib
else
  ifeq ($(OS_SIZE),32)
    CUDA_LIB_PATH  ?= $(CUDA_PATH)/lib
  else
    CUDA_LIB_PATH  ?= $(CUDA_PATH)/lib64
  endif
endif
# Common binaries
NVCC            ?= $(CUDA_BIN_PATH)/nvcc  
# Extra user flags
EXTRA_NVCCFLAGS ?=
EXTRA_LDFLAGS   ?=
EXTRA_CCFLAGS   ?= 
#`fltk-config --use-forms --use-gl --use-images --ldflags`
# CUDA code generation flags
#NO SUPPORT for SM1.X
#GENCODE_SM20    := -gencode arch=compute_20,code=sm_20
#GENCODE_SM30    := -gencode arch=compute_30,code=sm_30
#GENCODE_SM35    := -gencode arch=compute_35,code=sm_35
#GENCODE_FLAGS   :=  $(GENCODE_SM20) --ptxas-options=-v
#  -Xptxas -dlcm=cg
# --maxrregcount=63
# -use_fast_math -Xptxas -dlcm=cg  
# OS-specific build flags
ifneq ($(DARWIN),) 
      LDFLAGS   := -Xlinker -rpath $(CUDA_LIB_PATH) -L$(CUDA_LIB_PATH) -lcudart 
      CCFLAGS   := -arch $(OS_ARCH) 
else
  ifeq ($(OS_SIZE),32)
      LDFLAGS   := -L$(CUDA_LIB_PATH) -fopenmp -lcudart -lSDL2 -fopenmp -lpng -lSDL2_ttf
 #     CCFLAGS   := -m32 -O3
      CCFLAGS   := -O3 
  else
      LDFLAGS   := -L$(CUDA_LIB_PATH) -fopenmp -lcudart -lSDL2 -fopenmp -lpng -lSDL2_ttf
#      CCFLAGS   := -m64 -O3
      CCFLAGS   :=  -O3 -fopenmp 
  endif
endif
# OS-architecture specific flags
ifeq ($(OS_SIZE),32)
      NVCCFLAGS := -m32 -O3
else
      NVCCFLAGS := -m64 -O3 
endif
# Debug build flags
ifeq ($(strip $(DEBUG)), yes)
      CCFLAGS   += -g
      NVCCFLAGS += -g -G
      TARGET := debug
else
      TARGET := release
endif




############################################################################################################
INCLUDES := -I$(CUDA_INC_PATH)/ -I.  -I./include/  $(GENERICFLAG) 




INC_LIBS := -lcudart -lSDL2 -lpng -lSDL2_ttf -lcurand -fopenmp



PROJECTNAME=gross
OBJDIR=obj/
COMPILER_OPTIONS :=   -MMD -MP 
CUDA_COMPILER_OPTIONS := $(foreach option, $(COMPILER_OPTIONS), --compiler-options=$(option))


############################################################################################################
############################################################################################################
all: directories $(PROJECTNAME)
############################################################################################################
############################################################################################################
SRCDIR := src/

OBJS := ../main.o timer.o constants.o calculator.o  norm.o \
		RK1.o RK2.o RK3.o init.o draw.o savepng.o gnuplot.o device.o sdl.o


INCS:= include/
INCS:= 
CUDAOBJS  := $(patsubst %.o,$(OBJDIR)%.o,$(OBJS))

deps = $(CUDAOBJS:.o=.d)
############################################################################################################
############################################################################################################
$(OBJDIR)%.o: $(SRCDIR)%.cu 
	@echo "######################### Compiling: "$<" #########################"
	$(VERBOSE)mkdir -p $(dir $(abspath $@ ) )
	$(VERBOSE)$(NVCC) $(NVCCFLAGS) $(GENCODE_FLAGS) $(EXTRA_NVCCFLAGS) $(INCLUDES) -M $< -o ${@:.o=.d} \
		-odir $(@D)
	$(VERBOSE)$(NVCC) $(NVCCFLAGS) $(GENCODE_FLAGS) $(EXTRA_NVCCFLAGS) $(INCLUDES) -dc $< -o $@

$(OBJDIR)%.o: $(SRCDIR)%.cpp
	@echo "######################### Compiling: "$<" #########################"
	$(VERBOSE)$(GCC) $(CCFLAGS) $(LDFLAGS) $(INCLUDES) -MMD -MP  -c $< -o $@ 
$(OBJDIR)%.o: $(SRCDIR)%.c
	@echo "######################### Compiling: "$<" #########################"
	$(VERBOSE)$(GCC) $(CCFLAGS) $(LDFLAGS) $(INCLUDES) -MMD -MP  -c $< -o $@ 
############################################################################################################
############################################################################################################CUDAOBJ=$
$(OBJDIR)dlink.o: $(CUDAOBJS)
	@echo ""######################### Creating: "$(OBJDIR)/$(OBJDIR)dlink.o" #########################"
	$(VERBOSE)$(NVCC) $(NVCCFLAGS) $(GENCODE_FLAGS) -dlink -o $@ $^ -lcudavert
############################################################################################################
$(PROJECTNAME): $(MAINOBJ) $(CUDAOBJS) $(OBJDIR)dlink.o
	@echo "######################### Creating: "$(PROJECTNAME)" #########################"
#	$(EXEC) $(NVCC) $(ALL_LDFLAGS) $(GENCODE_FLAGS) -o $@ $+ $(LDFLAGS)
	$(VERBOSE)$(EXEC) $(GCC) $(CCFLAGS) $(LDFLAGS)  -o $@ $+ $(EXTRA_CCFLAGS)  $(INC_LIBS)

############################################################################################################
############################################################################################################
directories:
	$(VERBOSE)mkdir -p $(OBJDIR)
clean:
	$(VERBOSE)rm -r -f $(OBJDIR)
	$(VERBOSE)rm -f $(PROJECTNAME)  $(OBJDIR)$(deps)

pack: 
	@echo Generating Package gross_$(VERSION).tar.gz
#	@tar cvfz gross_$(VERSION).tar.gz *.cpp $(INCS)*  $(SRCDIR)* Makefile
	@tar cvfz gross_$(VERSION).tar.gz *.cpp src/* include/* $(INCS) Makefile 
	@echo Generated Package gross_$(VERSION).tar.gz

.PHONY : clean pack directories $(PROJECTNAME)

-include $(deps)
