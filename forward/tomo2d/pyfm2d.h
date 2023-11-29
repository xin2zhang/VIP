extern void c_fm2d(int *, double *, double *, int *, double *, double *,
        int *, int *, int *, double *, double *, double *, double *,
        int *, int *, int *, int *, double *, double *, double *, double *);

extern void c_fm2d_parallel(int *, double *, double *, int *, double *, double *,
        int *, int *, int *, double *, double *, double *, double *,
        int *, int *, int *, int *, int*, double *, double *, double *);

extern void c_fm2d_lglike(int *, double *, double *, int *, double *, double *,
        int *, int *, int *, double *, double *, double *, double *,
        int *, int *, int *, int *, int*, double *, double *, double *, double *, double *);

extern  void c_many_fm2d(int *nsrc, double *srcx, double *srcy, int *nrec, double *recx, double *recy,
               int *nx, int *ny, double *xmin, double *ymin, double *dx, double *dy,
               int *gdx, int *gdy, int *sdx, int *sext, int *nv, double *vel, double *tobs, 
               int *mask, double *res, double *grads, double *earth);
