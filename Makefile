pixelhasher: pixelhasher.cu comms.o config.o keccak2.h len.h json.o misc.o
	nvcc -Xptxas -O3,-v,-allow-expensive-optimizations=true -o pixelhasher comms.o config.o json.o misc.o pixelhasher.cu
comms.o: comms.cu comms.h len.h config.o json.o
	nvcc --lib -o comms.o config.o json.o comms.cu
comms_test: comms_test.cu comms.o
	nvcc -o comms_test comms.o config.o json.o misc.o comms_test.cu
config.o: config.cu config.h json.o
	nvcc --lib -o config.o json.o config.cu
config_test: config_test.cu config.o json.o
	nvcc -o config_test config.o json.o config_test.cu
json.o: json.cu json.h
	nvcc --lib -o json.o json.cu
json_test: json_test.cu json.o
	nvcc -o json_test json.o json_test.cu
misc.o: misc.cu misc.h
	nvcc --lib -o misc.o misc.cu
misc_test: misc_test.cu misc.o
	nvcc -o misc_test misc.o misc_test.cu
tags: *.cu *.h
	ctags *.cu *.h
clean:
	rm -f pixelhasher comms.o comms_test config.o config_test json.o json_test misc.o misc_test
