SDIR=src
ODIR=obj
BDIR=bin

CC=icc
CFLAGS=-Wall -O3 -I$(SDIR)

LIBS=-lm

_DEPS=data.h mem.h
DEPS=$(patsubst %,$(SDIR)/%,$(_DEPS))

_OBJ=data.o mem.o
OBJ=$(patsubst %,$(ODIR)/%,$(_OBJ))

$(ODIR)/%.o: $(SDIR)/%.c $(DEPS)
	mkdir -p $(ODIR) && $(CC) -c -o $@ $< $(CFLAGS) $(LIBS)

omp: $(SDIR)/omp.c $(OBJ)
	mkdir -p $(BDIR) && $(CC) -o $(BDIR)/$@ $^ $(CFLAGS) -qopenmp $(LIBS) && cp -R data bin

seq: $(SDIR)/seq.c $(OBJ)
	mkdir -p $(BDIR) && $(CC) -o $(BDIR)/$@ $^ $(CFLAGS) $(LIBS) && cp -R data bin

.PHONY: clean

clean:
	rm -f $(ODIR)/*.o $(BDIR)/*
