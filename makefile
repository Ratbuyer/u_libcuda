sm_version=90a


sparse:
	nvcc -arch=sm_${sm_version} -O0 examples/sparse.cu -lcudart -lcuda -o bins/bin

overlap:
	nvcc -arch=sm_${sm_version} -O0 examples/overlap.cu -lcudart -lcuda -o bins/bin

dense:
	nvcc -arch=sm_${sm_version} -O0 examples/dense.cu -lcudart -lcuda -o bins/bin

ptx:
	nvcc -arch=sm_${sm_version} -O0 examples/overlap.cu -lcudart -lcuda -o bin.ptx -ptx

run:
	./bins/bin

clean:
	rm -rf bins/*