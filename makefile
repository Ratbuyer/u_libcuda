sm_version=90a

test:
	nvcc -arch=sm_${sm_version} -O0 examples/test.cu -lcudart -lcuda -o bins/bin


run:
	./bins/bin


clean:
	rm -rf bins/*