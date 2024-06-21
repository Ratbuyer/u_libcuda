sm_version=90a
NVCC=/usr/local/cuda-12.3/bin/nvcc


sparse:
	${NVCC} -arch=sm_${sm_version} -O0 examples/sparse.cu -lcudart -lcuda -o bins/bin

overlap:
	${NVCC} -arch=sm_${sm_version} -O0 examples/overlap.cu -lcudart -lcuda -o bins/bin

dense:
	${NVCC} -arch=sm_${sm_version} -O0 examples/dense.cu -lcudart -lcuda -o bins/bin

ptx:
	${NVCC} -arch=sm_${sm_version} -O0 examples/overlap.cu -lcudart -lcuda -o bin.ptx -ptx

test:
	${NVCC} -arch=sm_${sm_version} -O0 examples/test.cu -lcudart -lcuda -o bins/bin

bin:
	${NVCC} -arch=sm_${sm_version} -O0 bin.ptx -cubin -o bins/bin
run:
	./bins/bin

clean:
	rm -rf bins/*