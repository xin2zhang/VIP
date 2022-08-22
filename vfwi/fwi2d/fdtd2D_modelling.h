// extern "C"
struct Source
{
	int s_iz, s_ix, r_iz, *r_ix, r_n;
	int *r_id;
};


// void model_2D(char input_file[200], float *vel_inner);
void getrik(int nt, float dt, float f0, float *rik);
void forward_aco_2D(int is, int nt, int ntx, int ntz, int ntp, int nx, int nz, 
					int Lc, int pml, int rnmax, int nt_interval,
					float dx, float dz, float dt, float f0, float velp_max, float velpaverage, 
					float *w, float *t11, float *t12, float *t13, float *fd, float *rik, float *velp, 
					float *p0, float *p1, float *p2, float *psave, float *record, struct Source ss[]);
void backward_aco_2D(int is, int nt, int ntx, int ntz, int ntp, int nx, int nz, 
					int Lc, int pml, int rnmax, int nt_interval,
					float dx, float dz, float dt, float f0, float velp_max, float velpaverage, 
					float *w, float *t11, float *t12, float *t13, float *fd, float *rik, float *velp, 
					float *p0, float *p1, float *p2, float *psave, float *record,
					float *grad, struct Source ss[]);
void fdtd_2d_calculate_p(int ntx, int ntz, int Lc, int pml, float dx, float dz, float dt, 
					float *fd, float *velp, float *p0, float *p1, float *p2, 
					float *w, float *t11, float *t12, float *t13);
void wavefield_IO(int forward_or_backward, int ntx, int ntz, int pml, int nt, int nt_interval, int it, 
				float dx, float dz, float dt, float *rik, float *velp, 
				float *p0, float *p1, float *p2, float *psave, float *record, float *grad,
				int s_ix, int s_iz, int r_iz, int *r_ix, int r_n);
void forward_IO(int ntx, int ntz, int pml, int nt, int nt_interval, int it, 
				float dx, float dz, float dt, float *rik, float *velp, 
				float *p0, float *p1, float *p2, float *psave, float *record,
				int s_ix, int s_iz, int r_iz, int *r_ix, int r_n);
void backward_IO(int ntx, int ntz, int pml, int nt, int nt_interval, int it, 
				float dx, float dz, float dt, float *velp, 
				float *p0, float *p1, float *p2, float *psave, float *record, float *grad,
				int s_ix, int s_iz, int r_iz, int *r_ix, int r_n);
void updata_p(int ntx, int ntz, float *p0, float *p1, float *p2);

void wavefield_initialization(int ntx, int ntz, float *p0, float *p1, float *p2);
void fd_coefficient(int Lc, float *fd);				
void get_velp(int pml, int ntx, int ntz, float *vel_inner, float *velp);
void read_parameters(char inputfile[200], int *nx, int *nz, int *pml0, int *Lc, int *ns, int *nt, int *ds, 
						int *ns0, int *depths, int *depthr, int *dr, int *nt_interval,
						float *dx, float *dz, float *dt, float *f0);
void read_int_value(char strtmp[256], FILE *fp, int *param);
void read_float_value(char strtmp[256], FILE *fp, float *param);


