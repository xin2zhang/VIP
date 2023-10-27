# List of source file
OBJ = fm2d_globalp.o fm2d_ttime.o fm2dray.o fm2d_wrapper.o 
F90 = gfortran
LIKE2D_LIB = libfmm2d.a
AR = ar -r
FFLAGS += -fPIC -fopenmp
#FFLAGS += -O0 -g -Wall -Wtabs -Wextra -Wconversion -fimplicit-none -fbacktrace -fcheck=all -ffpe-trap=zero,overflow,underflow -finit-real=nan 

# rule for building foward modeling code
$(LIKE2D_LIB): $(OBJ)
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
