sm_version=90a
NVCC=nvcc

all:
	make gemm
	make run

sparse:
	${NVCC} -arch=sm_${sm_version} -O0 examples/sparse.cu -lcudart -lcuda -o bins/bin

overlap:
	${NVCC} -arch=sm_${sm_version} -O0 examples/overlap.cu -lcudart -lcuda -o bins/bin

dense:
	${NVCC} -arch=sm_${sm_version} -O0 examples/dense.cu -lcudart -lcuda -o bins/bin

test:
	${NVCC} -arch=sm_${sm_version} -O0 examples/test.cu -lcudart -lcuda -o bins/bin

gemm:
	${NVCC} -arch=sm_${sm_version} -O0 examples/gemm.cu -lcudart -lcuda -o bins/bin





run:
	./bins/bin

clean:
	rm -rf bins/*