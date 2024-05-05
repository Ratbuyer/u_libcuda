sm_version=90a

test:
	nvcc -arch=sm_${sm_version} -O3 examples/sparse.cu -lcudart -lcuda -o bins/bin

ptx:
	nvcc -arch=sm_90a -O0 examples/test.cu -lcudart -lcuda -o bin.ptx -ptx

bin:
	nvcc -arch=sm_90a -cubin bin.ptx -o bins/bin

run:
	./bins/bin


clean:
	rm -rf bins/*