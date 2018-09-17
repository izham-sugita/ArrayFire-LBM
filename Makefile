CC = g++
NVCC = nvcc
RM = /bin/rm
PROG = LBMHeatEq

OBJS = AFLBMHeat2D.o LBMHeatEqKernel.o timestamp.o
#CFLAGS = -O3 -lafcuda
CFLAGS = -std=c++11 -O3 -lafcuda
LDFLAGS = ${CFLAGS}

M_ARCH = $(shell uname -m)

# for Device Code
CUDA_PATH = /usr/local/cuda-8.0
ifeq ($(M_ARCH), x86_64)
LDFLAGS += -L${CUDA_PATH}/lib64
else
LDFLAGS += -L${CUDA_PATH}/lib
endif
LDFLAGS += -lcudart -lcuda -lm
NFLAG += ${CFLAGS} -I /usr/local/cuda-8.0/include

#NFLAG += -maxrregcount 32

all : ${PROG}

${PROG} : ${OBJS}
	${NVCC} -o $@ ${OBJS} ${LDFLAGS}

LBMHeatEqKernel.o : LBMHeatEqKernel.cu
git	${NVCC} -c ${NFLAG} $<

AFLBMHeatEq2D.o : AFLBMHeatEq2D.cpp
	${CC} -c ${LDFLAGS} $<

timestamp.o : timestamp.cpp
	${CC} -c ${LDFLAGS} $<

clean :
	${RM} -f ${PROG} ${OBJS} *.o 



