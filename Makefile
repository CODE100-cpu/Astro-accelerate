NVCC = nvcc
ARCH = -arch=sm_86
SRC = aa_host_rfi.cu
OUT = aa_host_rfi
LIBS = -lcurand -lcublas
NVCCFLAGS = -g -O0 -lineinfo -Xptxas -O0
all: $(OUT)

$(OUT): $(SRC)
	$(NVCC) $(NVCCFLAGS) $(ARCH) $(SRC) -o $(OUT) $(LIBS)

clean:
	rm -f $(OUT)