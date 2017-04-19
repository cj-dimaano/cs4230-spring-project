SDIR=src
ODIR=obj
BDIR=bin

PCC=nvcc
CC=gcc
PCFLAGS=-I$(SDIR) -I/usr/local/cuda/8.0/cuda/include
CFLAGS=-I$(SDIR)

LIBS=-lm

_DEPS=data.h mem.h
DEPS=$(patsubst %,$(SDIR)/%,$(_DEPS))

_OBJ=data.o mem.o
OBJ=$(patsubst %,$(ODIR)/%,$(_OBJ))

$(ODIR)/%.o: $(SDIR)/%.c $(DEPS)
	mkdir -p $(ODIR) && $(CC) -c -o $@ $< $(CFLAGS) $(LIBS)

cuda: $(SDIR)/cuda.cu $(OBJ)
	mkdir -p $(BDIR) && $(PCC) -o $(BDIR)/$@ $^ $(PCFLAGS) $(LIBS) && cp -R data bin

seq: $(SDIR)/seq.c $(OBJ)
	mkdir -p $(BDIR) && $(CC) -o $(BDIR)/$@ $^ $(CFLAGS) $(LIBS) && cp -R data bin

.PHONY: clean

clean:
	rm -f $(ODIR)/*.o $(BDIR)/*
