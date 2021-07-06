.PHONY: all build test clean

all: build
	CC=g++ LDSHARED='$(shell python3 scripts/configure.py)' python3 setup.py build
	

build:
	nvcc -rdc=true --compiler-options '-fPIC' -c -o temp.o voxel_util.cu
	nvcc -dlink --compiler-options '-fPIC' -o voxel_util.o temp.o -lcudart
	rm -f libvoxelutil.a
	ar cru libvoxelutil.a voxel_util.o temp.o
	ranlib libvoxelutil.a
	rm temp.o voxel_util.o
	


clean:
	rm -f libvoxelutil.a *.o main temp.py
	rm -rf build
