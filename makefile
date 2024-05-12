sm_version=90a

test:
	nvcc -arch=sm_${sm_version} -O0 examples/overlap.cu -lcudart -lcuda -o bins/bin

dense:
	nvcc -arch=sm_${sm_version} -O0 examples/dense.cu -lcudart -lcuda -o bins/bin

ptx:
	nvcc -arch=sm_${sm_version} -O0 examples/dense.cu -lcudart -lcuda -o bin.ptx -ptx

bin:
	nvcc -arch=sm_${sm_version} -cubin bin.ptx -o bins/bin

run:
	./bins/bin


clean:
	rm -rf bins/*