# List of source file
OBJ = utils.o transform.o kernel.o 
F90 = gfortran
LIB = libsvgd.a
AR = ar -r
FFLAGS += -I/usr/include -fPIC -O3 -funroll-loops -ffast-math -fopenmp
#FFLAGS += -O0 -g -Wall -Wtabs -Wextra -Wconversion -fimplicit-none -fbacktrace -fcheck=all -ffpe-trap=zero,overflow,underflow -finit-real=nan 

# rule for building foward modeling code
$(LIB): $(OBJ)
	$(AR) $@ $^

# rule for building object file
%.o: %.f90
	$(F90) $(FFLAGS) -c $< -o $@

%.o: %.cpp
	$(CC) $(CFLAGS) -c $< -o $@

.PHONY: clean cleanall

clean:
	rm -f *.o *.mod

cleanall: clean
	rm -f *.a
