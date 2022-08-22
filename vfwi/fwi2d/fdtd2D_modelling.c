//========2D acoustic wave modeling FDM===//
//========2D acoustic wave modeling FDM===//
#define _DEFAULT_SOURCE
#include "stdio.h"
#include "stdlib.h"
#include "math.h"
#include <time.h>
#include <sys/time.h>
#include <string.h>
#include "fdtd2D_modelling.h"
#include "omp.h"
// using namespace std;

#define pi 3.1415926


void fwi_2D(char input_file[200], float *vel_inner, float *record_syn, 
				float *record_obs, float *grad, int verbose)
{
  	//============Time begin====================//
  	//============Time begin====================//
	// clock_t start,finish;	// CPU time
	// start=clock();	
	
	
	time_t begin;
	// time(&begin);
    //struct timespec begin_time, end_time;
	if(verbose)
	{		
        //clock_gettime(CLOCK_REALTIME, &begin_time);		//Elapsed time

		printf("==============================================\n");
        //printf("Program starting time: %s",ctime(&begin));
		printf("\n");

		printf("The number of thread available: %d\n", omp_get_max_threads());
	}

	//==============model parameters=============//
	//==============model parameters=============//
	int nx;//=200;
	int nz;//=200;
  	// int rnmax=0;				// maxium receiver numbers
	int pml0;//=10;
	int Lc;// = 3+1;
	int ns;//=1;
	int nt;//=1001;
	int ds;//=10;
	int ns0;//=100;				//the distance of first shot to the left side of the model
	int depths;//=nz/2;//100;			//shot and receiver depth
	int depthr;//=nz/2;
  	int dr;//=1;			//receivers distributed in every two points in both X- and Y-directions
	int nt_interval;//=1;
	
	float dx;//=10.0;
	float dz;//=10.0;
	float dt;//=0.001;
	float f0;//=20.0;
	
	// char input_file[200] = "./input/input_param.txt";

	read_parameters(input_file, &nx, &nz, &pml0, &Lc, &ns, &nt, &ds, &ns0, 
						&depths, &depthr, &dr, &nt_interval,
						&dx, &dz, &dt, &f0);
	Lc += 1;	// second order FD scheme +1 
	// int ntsave = nt/nt_interval;

	int it,ix,iz,ir,i,ip;
	int pml = pml0+Lc;
	int ntz=nz+2*pml;			// total vertical samples 
	int ntx=nx+2*pml;			// total horizontal samples 
	
	int ntp=ntz*ntx;		// total samples of the model
	int np=nx*nz;			// total samples of the simulating domain
	
	char filename[200];
	FILE *fp;

	if(verbose)
	{	
		printf("ntx=%d, ntz=%d\n",ntx, ntz);		
		printf("nx=%d,  nz=%d,  nt=%d,  ns=%d\n",nx, nz, nt, ns);		
		printf("dx=%f,  dz=%f,  dt=%f\n",dx, dz, dt);		
		printf("==============================================\n");
	}
	//================wavelet definition============//
	//================wavelet definition============//
	float *rik;			// ricker wavelet
	rik=(float *) malloc(nt*sizeof(float));
	getrik(nt, dt, f0, rik);

	float *fd;	//finite difference coefficients
	fd = (float *)malloc(sizeof(float)*Lc);
	fd_coefficient(Lc, fd);
	
	
	//==============observation system definition===============//
	//==============observation system definition===============//
  	int is;  	
  	int rnmax=nx/dr;			//maximum receiver number along X-direction
  	
	struct Source ss[ns];			// struct pointer for source variables
	
	for(is=0; is<ns; is++)
	{
		ss[is].s_ix=is*ds+pml+ns0;	// shot horizontal position
		ss[is].s_iz=pml+depths;		// shot vertical position
		ss[is].r_iz=pml+depthr;		// receiver vertical position
		
		ss[is].r_n =rnmax;	// receiver number for each shot
					
		if(rnmax<ss[is].r_n)
			rnmax=ss[is].r_n;		// maxium receiver number
	}
	
	for(is=0; is<ns; is++)
	{
		ss[is].r_ix=(int*)malloc(sizeof(int)*ss[is].r_n);
		memset(ss[is].r_ix, 0, sizeof(int)*ss[is].r_n);	
		
		ss[is].r_id=(int*)malloc(sizeof(int)*ntx);	// the receiver ID num corresponding to the grid point ix
		memset(ss[is].r_id, -1, sizeof(int)*ntx);	
		//r_id[] !=-1 means this poing (ix) has one receiver
		//r_id[] ==-1 means this poing (ix) does not have receiver
	}

	// #pragma omp parallel for collapse(2) private(is,ip) 
	for(is=0;is<ns;is++)
	{
		for(ip=0;ip<ss[is].r_n;ip++)
		{
			ss[is].r_ix[ip] = pml+ip*dr;
			ss[is].r_id[ss[is].r_ix[ip]]=ip;			
		}
	}

	if(verbose)
	{	
		printf("The total shot number is %d\n",ns);
		printf("The maximum trace number for source is %d\n",rnmax);
		printf("\n==============================================\n");	
	}

	//===================wavefield and record=============//
	//===================wavefield and record=============//
	float *p0, *p1, *p2;
	float *psave;
	int nt_save = nt/nt_interval + 1;
	if(nt%nt_interval==0)
		nt_save -= 1;

	p0 = (float*)malloc(sizeof(float)*ntp);
	p1 = (float*)malloc(sizeof(float)*ntp);
	p2 = (float*)malloc(sizeof(float)*ntp);
	psave = (float*)malloc(sizeof(float)*ntp*nt_save);

	float *record;
	record = (float*)malloc(sizeof(float)*nt*rnmax);

	memset(p0, 0, ntp*sizeof(float));
	memset(p1, 0, ntp*sizeof(float));
	memset(p2, 0, ntp*sizeof(float));
	memset(psave, 0, ntp*nt_save*sizeof(float));
	memset(record, 0, nt*rnmax*sizeof(float));


	//===================velocity and Q models=============//
	//===================velocity and Q models=============//
	float *velp;
	float velp_max, velp_min;
	float velpaverage;
	velp = (float*)malloc(sizeof(float)*ntp);
	
	get_velp(pml, ntx, ntz, vel_inner, velp);	

	velpaverage=0.0;
	for(ip=0;ip<ntp;ip++)
	{
		velpaverage+=velp[ip];
	}
	velpaverage/=ntp;

	velp_max=0.0;
	velp_min=6000.0;
	for(ip=0;ip<ntp;ip++)
	{
		if(velp[ip]>=velp_max){velp_max=velp[ip];}
		if(velp[ip]<=velp_min){velp_min=velp[ip];}
	}
	
	if(verbose)
	{	
		printf("velp_max = %f\n",velp_max);
		printf("velp_min = %f\n",velp_min); 		
		printf("velpaverage=%lf\n",velpaverage);
		printf("==============================================\n");	
	}

	//==============Absorbing Boundary Condition=============//
	//==============Absorbing Boundary Condition=============//
	float *w, *t11, *t12, *t13;
	float s, alpha=1.0;
	w = (float *)malloc(sizeof(float)*pml0);
	t11 = (float *)malloc(sizeof(float)*ntp);
	t12 = (float *)malloc(sizeof(float)*ntp);
	t13 = (float *)malloc(sizeof(float)*ntp);

	for(ix=0;ix<pml0;ix++)
	{
		w[ix] = 1.0-1.0*ix/pml0;
	}

	#pragma omp parallel for collapse(2) private(iz,ix,s)
	for(ix=0;ix<ntx;ix++)
		for(iz=0;iz<ntz;iz++)
		{
			s = alpha*velp[iz*ntx+ix]*dt/dx;
			t11[iz*ntx+ix] = (2-s)*(1-s)/2;
			t12[iz*ntx+ix] = s*(2-s);
			t13[iz*ntx+ix]=s*(s-1)/2;
		}

	if(verbose)
	{	
		printf("\n==============================================\n");	
		printf("FWI begin!\n");			
		printf("\n==============================================\n");	
	}

	for(is=0; is<ns; is++)	//
	{
		//============forward simulation==========//
		//============forward simulation==========//

		forward_aco_2D
		(
			is, nt, ntx, ntz, ntp, nx, nz, Lc, pml, rnmax, nt_interval,
			dx, dz, dt, f0, velp_max, velpaverage, w, t11, t12, t13,
			fd, rik, velp, p0, p1, p2, psave, record, ss
		);

		// sprintf(filename,"./output/record_aco_shot%d.bin",is);
		// fp=fopen(filename,"wb");
		// for(it=0;it<nt;it++)
		// 	for(ir=0; ir<ss[is].r_n; ir++)
		// 		fwrite(&record[it*ss[is].r_n+ir],sizeof(float),1,fp);
		// fclose(fp);

		//============record synthetic data==================//
		#pragma omp parallel for collapse(2) private(it,ir)
		for(it=0;it<nt;it++)
			for(ir=0; ir<ss[is].r_n; ir++)
			{
				record_syn[is*nt*ss[is].r_n+it*ss[is].r_n+ir] = record[it*ss[is].r_n+ir];
				record[it*ss[is].r_n+ir] -= record_obs[is*nt*ss[is].r_n+it*ss[is].r_n+ir];
			}

		backward_aco_2D
		(
			is, nt, ntx, ntz, ntp, nx, nz, Lc, pml, rnmax, nt_interval,
			dx, dz, dt, f0, velp_max, velpaverage, w, t11, t12, t13,
			fd, rik, velp, p0, p1, p2, psave, record, grad, ss
		);

		// sprintf(filename,"./output/grad_aco_shot%d.bin",is);
		// fp=fopen(filename,"wb");
		// for(iz=0;iz<nz;iz++)
		// 	for(ix=0; ix<nx; ix++)
		// 		fwrite(&grad[iz*nx+ix],sizeof(float),1,fp);
		// fclose(fp);

		if(verbose)
			printf("Gradient computation finished: is=%2d\n", is);
	}

	if(verbose)
	{		
		printf("\n==============================================\n");	
		printf("FWI end!\n");	
		printf("\n==============================================\n");	
	}	

	//===============free variables=====================//
	//===============free variables=====================//
	
	// cuda_Device_free(plan, GPU_N);
	
	for(is=0;is<ns;is++)
	{
		free(ss[is].r_ix);
		free(ss[is].r_id);
	} 
	
	free(rik);
	free(fd);
	free(velp);
	free(p0);
	free(p1);
	free(p2);
	free(psave);
	free(w);
	free(t11);free(t12);free(t13);
	free(record);

	// finish=clock();
	// time(&end);
	// time_t elapsed = end - begin;	
	// time=(float)(finish-start)/CLOCKS_PER_SEC;	

	if(verbose)
	{	
		//clock_gettime(CLOCK_REALTIME, &end_time);
		//long seconds = end_time.tv_sec - begin_time.tv_sec;
		//long nanoseconds = end_time.tv_nsec - begin_time.tv_nsec;
		//double elapsed_time = seconds + nanoseconds*1e-9;
		
		printf("====================================\n");		
		// printf("The running time is %ldseconds\n", elapsed);	
		//printf("The running time is %fseconds\n", elapsed_time);	
		printf("====================================\n");		
		printf("\n");	
		printf("The program has been finished\n");	
		printf("====================================\n");		
		printf("\n");
	}
	// MPI_Barrier(comm);
	// MPI_Finalize();	
	
	// return 0;
}


void forward_2D(char input_file[200], float *vel_inner, float *record_syn, int verbose)
{
  	//============Time begin====================//
  	//============Time begin====================//
	// clock_t start,finish;	// CPU time
	// start=clock();	


	//time_t begin;
	// time(&begin);
	//begin struct timespec begin_time, end_time;
	if(verbose)
	{	
		//clock_gettime(CLOCK_REALTIME, &begin_time);		//Elapsed time

		printf("==============================================\n");
		//printf("Program starting time: %s",ctime(&begin));
		printf("\n");	

		printf("The number of thread available: %d\n", omp_get_max_threads());
	}

	//==============model parameters=============//
	//==============model parameters=============//
	int nx;//=200;
	int nz;//=200;
  	// int rnmax=0;				// maxium receiver numbers
	int pml0;//=10;
	int Lc;// = 3+1;
	int ns;//=1;
	int nt;//=1001;
	int ds;//=10;
	int ns0;//=100;				//the distance of first shot to the left side of the model
	int depths;//=nz/2;//100;			//shot and receiver depth
	int depthr;//=nz/2;
  	int dr;//=1;			//receivers distributed in every two points in both X- and Y-directions
	int nt_interval;//=1;
	
	float dx;//=10.0;
	float dz;//=10.0;
	float dt;//=0.001;
	float f0;//=20.0;
	
	// char input_file[200] = "./input/input_param.txt";

	read_parameters(input_file, &nx, &nz, &pml0, &Lc, &ns, &nt, &ds, &ns0, 
						&depths, &depthr, &dr, &nt_interval,
						&dx, &dz, &dt, &f0);
	Lc += 1;	// second order FD scheme +1 
	// int ntsave = nt/nt_interval;

	int it,ix,iz,ir,i,ip;
	int pml = pml0+Lc;
	int ntz=nz+2*pml;			// total vertical samples 
	int ntx=nx+2*pml;			// total horizontal samples 
	
	int ntp=ntz*ntx;		// total samples of the model
	int np=nx*nz;			// total samples of the simulating domain
	
	char filename[200];
	FILE *fp;

	if(verbose)
	{	
		printf("ntx=%d, ntz=%d\n",ntx, ntz);		
		printf("nx=%d,  nz=%d,  nt=%d,  ns=%d\n",nx, nz, nt, ns);		
		printf("dx=%f,  dz=%f,  dt=%f\n",dx, dz, dt);		
		printf("==============================================\n");
	}

	//================wavelet definition============//
	//================wavelet definition============//
	float *rik;			// ricker wavelet
	rik=(float *) malloc(nt*sizeof(float));
	getrik(nt, dt, f0, rik);

	float *fd;	//finite difference coefficients
	fd = (float*)malloc(sizeof(float)*Lc);
	fd_coefficient(Lc, fd);
	
	
	//==============observation system definition===============//
	//==============observation system definition===============//
  	int is;  	
  	int rnmax=nx/dr;			//maximum receiver number along X-direction
  	
	struct Source ss[ns];			// struct pointer for source variables
	
	for(is=0; is<ns; is++)
	{
		ss[is].s_ix=is*ds+pml+ns0;	// shot horizontal position
		ss[is].s_iz=pml+depths;		// shot vertical position
		ss[is].r_iz=pml+depthr;		// receiver vertical position
		
		ss[is].r_n =rnmax;	// receiver number for each shot
					
		if(rnmax<ss[is].r_n)
			rnmax=ss[is].r_n;		// maxium receiver number
	}
	
	for(is=0; is<ns; is++)
	{
		ss[is].r_ix=(int*)malloc(sizeof(int)*ss[is].r_n);
		memset(ss[is].r_ix, 0, sizeof(int)*ss[is].r_n);	
		
		ss[is].r_id=(int*)malloc(sizeof(int)*ntx);	// the receiver ID num corresponding to the grid point ix
		memset(ss[is].r_id, -1, sizeof(int)*ntx);	
		//r_id[] !=-1 means this poing (ix) has one receiver
		//r_id[] ==-1 means this poing (ix) does not have receiver
	}

	// #pragma omp parallel for collapse(2) private(is,ip) 
	for(is=0;is<ns;is++)
	{
		for(ip=0;ip<ss[is].r_n;ip++)
		{
			ss[is].r_ix[ip] = pml+ip*dr;
			ss[is].r_id[ss[is].r_ix[ip]]=ip;			
		}
	}

	if(verbose)
	{
		printf("The total shot number is %d\n",ns);
		printf("The maximum trace number for source is %d\n",rnmax);
		printf("\n==============================================\n");	
	}

	//===================wavefield and record=============//
	//===================wavefield and record=============//
	float *p0, *p1, *p2;
	float *psave;
	int nt_save = nt/nt_interval + 1;
	if(nt%nt_interval==0)
		nt_save -= 1;

	p0 = (float*)malloc(sizeof(float)*ntp);
	p1 = (float*)malloc(sizeof(float)*ntp);
	p2 = (float*)malloc(sizeof(float)*ntp);
	psave = (float*)malloc(sizeof(float)*ntp*nt_save);

	float *record;
	record = (float*)malloc(sizeof(float)*nt*rnmax);

	memset(p0, 0, ntp*sizeof(float));
	memset(p1, 0, ntp*sizeof(float));
	memset(p2, 0, ntp*sizeof(float));
	memset(psave, 0, ntp*nt_save*sizeof(float));
	memset(record, 0, nt*rnmax*sizeof(float));
	// memset(record_syn, 0, ns*nt*rnmax*sizeof(float));


	//===================velocity and Q models=============//
	//===================velocity and Q models=============//
	float *velp;
	float velp_max, velp_min;
	float velpaverage;
	velp = (float*)malloc(sizeof(float)*ntp);
	
	get_velp(pml, ntx, ntz, vel_inner, velp);	

	velpaverage=0.0;
	for(ip=0;ip<ntp;ip++)
	{
		velpaverage+=velp[ip];
	}
	velpaverage/=ntp;

	velp_max=0.0;
	velp_min=6000.0;
	for(ip=0;ip<ntp;ip++)
	{
		if(velp[ip]>=velp_max){velp_max=velp[ip];}
		if(velp[ip]<=velp_min){velp_min=velp[ip];}
	}
	
	if(verbose)
	{	
		printf("velp_max = %f\n",velp_max);
		printf("velp_min = %f\n",velp_min); 		
		printf("velpaverage=%lf\n",velpaverage);
		printf("==============================================\n");	
	}


	//==============Absorbing Boundary Condition=============//
	//==============Absorbing Boundary Condition=============//
	float *w, *t11, *t12, *t13;
	float s, alpha=1.0;
	w = (float *)malloc(sizeof(float)*pml0);
	t11 = (float *)malloc(sizeof(float)*ntp);
	t12 = (float *)malloc(sizeof(float)*ntp);
	t13 = (float *)malloc(sizeof(float)*ntp);

	for(ix=0;ix<pml0;ix++)
	{
		w[ix] = 1.0-1.0*ix/pml0;
	}

	#pragma omp parallel for collapse(2) private(iz,ix,s)
	for(ix=0;ix<ntx;ix++)
		for(iz=0;iz<ntz;iz++)
		{
			s = alpha*velp[iz*ntx+ix]*dt/dx;
			t11[iz*ntx+ix] = (2-s)*(1-s)/2;
			t12[iz*ntx+ix] = s*(2-s);
			t13[iz*ntx+ix]=s*(s-1)/2;
		}

	if(verbose)
	{	
		printf("\n==============================================\n");	
		printf("Forward begin!\n");			
		printf("\n==============================================\n");	
	}

	for(is=0; is<ns; is++)	//
	{
		//============forward simulation==========//
		//============forward simulation==========//

		forward_aco_2D
		(
			is, nt, ntx, ntz, ntp, nx, nz, Lc, pml, rnmax, nt_interval,
			dx, dz, dt, f0, velp_max, velpaverage, w, t11, t12, t13,
			fd, rik, velp, p0, p1, p2, psave, record, ss
		);

		// sprintf(filename,"./output/record_aco_shot%d.bin",is);
		// fp=fopen(filename,"wb");
		// for(it=0;it<nt;it++)
		// 	for(ir=0; ir<ss[is].r_n; ir++)
		// 		fwrite(&record[it*ss[is].r_n+ir],sizeof(float),1,fp);
		// fclose(fp);

		//============record synthetic data==================//
		#pragma omp parallel for collapse(2) private(it,ir)
		for(it=0;it<nt;it++)
			for(ir=0; ir<ss[is].r_n; ir++)
				record_syn[is*nt*ss[is].r_n+it*ss[is].r_n+ir] = record[it*ss[is].r_n+ir];

		if(verbose)
			printf("Foward modelling finished: is=%2d\n", is);
	}

	if(verbose)
	{	
		printf("\n==============================================\n");	
		printf("Forward end!\n");	
		printf("\n==============================================\n");	
	}

	//===============free variables=====================//
	//===============free variables=====================//
	
	// cuda_Device_free(plan, GPU_N);
	
	for(is=0;is<ns;is++)
	{
		free(ss[is].r_ix);
		free(ss[is].r_id);
	} 
	
	free(rik);
	free(fd);
	free(velp);
	free(p0);
	free(p1);
	free(p2);
	free(psave);
	free(w);
	free(t11);free(t12);free(t13);
	free(record);


	// finish=clock();
	// time(&end);
	// time_t elapsed = end - begin;	
	// time=(float)(finish-start)/CLOCKS_PER_SEC;	

	if(verbose)
	{
		//clock_gettime(CLOCK_REALTIME, &end_time);
		//long seconds = end_time.tv_sec - begin_time.tv_sec;
		//long nanoseconds = end_time.tv_nsec - begin_time.tv_nsec;
		//double elapsed_time = seconds + nanoseconds*1e-9;
		
		printf("====================================\n");		
		// printf("The running time is %ldseconds\n", elapsed);	
		//printf("The running time is %fseconds\n", elapsed_time);	
		printf("====================================\n");		
		printf("\n");	
		printf("The program has been finished\n");	
		printf("====================================\n");		
		printf("\n");
	}
	// MPI_Barrier(comm);
	// MPI_Finalize();	
	
	// return 0;
}


void forward_aco_2D(int is, int nt, int ntx, int ntz, int ntp, int nx, int nz, 
					int Lc, int pml, int rnmax, int nt_interval,
					float dx, float dz, float dt, float f0, float velp_max, float velpaverage, 
					float *w, float *t11, float *t12, float *t13, float *fd, float *rik, float *velp, 
					float *p0, float *p1, float *p2, float *psave, float *record, struct Source ss[])
{
	int i, it, ix, iz, ir;
	int forward_or_backward = 1;
	wavefield_initialization(ntx, ntz, p0, p1, p2);

	char filename[100];

	for(it=0; it<nt; it++)
	{
		fdtd_2d_calculate_p(ntx, ntz, Lc, pml, dx, dz, dt, fd, velp, p0, p1, p2, w, t11, t12, t13);
		forward_IO(ntx, ntz, pml, nt, nt_interval, it, dx, dz, dt, rik, velp, p0, p1, p2, psave, record,
						ss[is].s_ix, ss[is].s_iz, ss[is].r_iz, ss[is].r_ix, ss[is].r_n);
		updata_p(ntx, ntz, p0, p1, p2);

		// if(it%100 == 0)
		// {
		// 	printf("forward propagation,is=%2d,it=%4d\n",is,it);

		// 	// sprintf(filename,"./output/forward_snapshot%d.bin",it);
		// 	// FILE *fp=fopen(filename,"wb");
		// 	// for(iz=0; iz<ntz; iz++)
		// 	// 	for(ix=0;ix<ntx;ix++)
		// 	// 		fwrite(&p2[iz*ntx+ix],sizeof(float),1,fp);
		// 	// fclose(fp);
		// }
	}
}


void backward_aco_2D(int is, int nt, int ntx, int ntz, int ntp, int nx, int nz, 
					int Lc, int pml, int rnmax, int nt_interval,
					float dx, float dz, float dt, float f0, float velp_max, float velpaverage, 
					float *w, float *t11, float *t12, float *t13, float *fd, float *rik, float *velp, 
					float *p0, float *p1, float *p2, float *psave, float *record,
					float *grad, struct Source ss[])
{
	int i, it, ix, iz, ir;
	int forward_or_backward = 2;
	wavefield_initialization(ntx, ntz, p0, p1, p2);

	char filename[100];

	for(it=nt-1; it>=0; it--)
	{
		fdtd_2d_calculate_p(ntx, ntz, Lc, pml, dx, dz, dt, fd, velp, p0, p1, p2, w, t11, t12, t13);
		backward_IO(ntx, ntz, pml, nt, nt_interval, it, dx, dz, dt, velp, p0, p1, p2, psave, record, grad,
						ss[is].s_ix, ss[is].s_iz, ss[is].r_iz, ss[is].r_ix, ss[is].r_n);
		updata_p(ntx, ntz, p0, p1, p2);

		// if(it%100 == 0)
		// {
		// 	printf("backward propagation,is=%2d,it=%4d\n",is,it);

		// 	// sprintf(filename,"./output/backward_snapshot%d.bin",it);
		// 	// FILE *fp=fopen(filename,"wb");
		// 	// for(iz=0; iz<ntz; iz++)
		// 	// 	for(ix=0;ix<ntx;ix++)
		// 	// 		fwrite(&p2[iz*ntx+ix],sizeof(float),1,fp);
		// 	// fclose(fp);
		// }
	}
}


void fdtd_2d_calculate_p(int ntx, int ntz, int Lc, int pml, float dx, float dz, float dt, 
					float *fd, float *velp, float *p0, float *p1, float *p2, 
					float *w, float *t11, float *t12, float *t13)
{
	int ix, iz, ic, ip;
	float d2p_dx2, d2p_dz2;

	// #pragma omp parallel for collapse(3) private(iz,ix, ic, d2p_dx2, d2p_dz2)
	#pragma omp parallel for collapse(2) private(iz,ix, ic, d2p_dx2, d2p_dz2) 
	for(ix=Lc; ix<ntx-Lc; ix++)
	{
		for(iz=Lc; iz<ntz-Lc; iz++)
		{
			d2p_dx2 = fd[0]*p1[iz*ntx+ix];
			for(ic=1; ic<Lc; ic++)
				d2p_dx2 += fd[ic]*(p1[iz*ntx+(ix+ic)]+p1[iz*ntx+(ix-ic)]);
			
			d2p_dz2 = fd[0]*p1[iz*ntx+ix];
			for(ic=1; ic<Lc; ic++)
				d2p_dz2 += fd[ic]*(p1[(iz+ic)*ntx+ix]+p1[(iz-ic)*ntx+ix]);

			p2[iz*ntx+ix] = 2*p1[iz*ntx+ix] - p0[iz*ntx+ix] + powf(velp[iz*ntx+ix]*dt, 2) * 
									(d2p_dx2/dx/dx + d2p_dz2/dz/dz);
		}
	}

	//=====ABC=======//
	#pragma omp parallel for collapse(2) private(iz,ix,ip)
	for(ix=0; ix<ntx; ix++)
	{
		for(iz=Lc;iz<pml;iz++)
		{
			ip = iz*ntx+ix;
			p2[ip] = w[iz-Lc]*(
								2*t11[ip]*p1[ip]+2*t12[ip]*p1[ip+ntx]+2*t13[ip]*p1[ip+2*ntx] -
								t11[ip]*t11[ip]*p0[ip]-2*t11[ip]*t12[ip]*p0[ip+ntx]-
								(t12[ip]*t12[ip]+2*t11[ip]*t13[ip])*p0[ip+2*ntx] - 
								2*t12[ip]*t13[ip]*p0[ip+3*ntx]-t13[ip]*t13[ip]*p0[ip+4*ntx]
							)
							+(1.0-w[iz-Lc])*p2[ip];
		}
	}

	#pragma omp parallel for collapse(2) private(iz,ix,ip)
	for(ix=0; ix<ntx; ix++)
	{	
		for(iz=ntz-pml; iz<ntz-Lc; iz++)
		{
			ip = iz*ntx+ix;
			p2[ip] = w[ntz-Lc-iz-1]*(
								2*t11[ip]*p1[ip]+2*t12[ip]*p1[ip-ntx]+2*t13[ip]*p1[ip-2*ntx] -
								t11[ip]*t11[ip]*p0[ip]-2*t11[ip]*t12[ip]*p0[ip-ntx]-
								(t12[ip]*t12[ip]+2*t11[ip]*t13[ip])*p0[ip-2*ntx] - 
								2*t12[ip]*t13[ip]*p0[ip-3*ntx]-t13[ip]*t13[ip]*p0[ip-4*ntx]
							)
							+(1.0-w[ntz-Lc-iz-1])*p2[ip];
		}
	}

	#pragma omp parallel for collapse(2) private(iz,ix,ip)
	for(iz=0;iz<ntz;iz++)
	{
		for(ix=Lc; ix<pml;ix++)
		{
			ip = iz*ntx+ix;
			p2[ip] = w[ix-Lc]*(
								2*t11[ip]*p1[ip]+2*t12[ip]*p1[ip+1]+2*t13[ip]*p1[ip+2*1] -
								t11[ip]*t11[ip]*p0[ip]-2*t11[ip]*t12[ip]*p0[ip+1]-
								(t12[ip]*t12[ip]+2*t11[ip]*t13[ip])*p0[ip+2*1] - 
								2*t12[ip]*t13[ip]*p0[ip+3*1]-t13[ip]*t13[ip]*p0[ip+4*1]
							)
							+(1.0-w[ix-Lc])*p2[ip];
		}
	}

	#pragma omp parallel for collapse(2) private(iz,ix,ip)
	for(iz=0;iz<ntz;iz++)
	{	
		for(ix=ntx-pml;ix<ntx-Lc;ix++)
		{
			ip = iz*ntx+ix;
			p2[ip] = w[ntx-Lc-ix-1]*(
								2*t11[ip]*p1[ip]+2*t12[ip]*p1[ip-1]+2*t13[ip]*p1[ip-2*1] -
								t11[ip]*t11[ip]*p0[ip]-2*t11[ip]*t12[ip]*p0[ip-1]-
								(t12[ip]*t12[ip]+2*t11[ip]*t13[ip])*p0[ip-2*1] - 
								2*t12[ip]*t13[ip]*p0[ip-3*1]-t13[ip]*t13[ip]*p0[ip-4*1]
							)
							+(1.0-w[ntx-Lc-ix-1])*p2[ip];
		}
	}
}


void forward_IO(int ntx, int ntz, int pml, int nt, int nt_interval, int it, 
				float dx, float dz, float dt, float *rik, float *velp, 
				float *p0, float *p1, float *p2, float *psave, float *record,
				int s_ix, int s_iz, int r_iz, int *r_ix, int r_n)
{
	int ix, iz, ip, ir;

	//==============Add Ricker wavelet=============//
	p2[s_iz*ntx+s_ix] += rik[it]*dt*dt*velp[s_iz*ntx+s_ix]*velp[s_iz*ntx+s_ix];

	//=======save forward wavefield to calculate gradient=========//
	if(it%nt_interval == 0)
	{
		#pragma omp parallel for collapse(2) private(iz, ix)
		for(ix=0;ix<ntx;ix++)
		{
			for(iz=0;iz<ntz;iz++)
			{
				psave[(int)(it/nt_interval)*ntx*ntz+iz*ntx+ix] =
						(p2[iz*ntx+ix]-2*p1[iz*ntx+ix]+p0[iz*ntx+ix])/dt/dt;
			}
		}
	}

	//==============Record seismograms=============//
	#pragma omp parallel for private(ir)
	for(ir=0; ir<r_n; ir++)
	{
		// ir = r_id[ix];
		// if(ir>=0)
		// {
		record[it*r_n+ir] = p2[r_iz*ntx+r_ix[ir]];
		// }
	}
}

void backward_IO(int ntx, int ntz, int pml, int nt, int nt_interval, int it, 
				float dx, float dz, float dt, float *velp, 
				float *p0, float *p1, float *p2, float *psave, float *record, float *grad,
				int s_ix, int s_iz, int r_iz, int *r_ix, int r_n)
{
	int ix, iz, ip, ir;

	//==============Back propagate the residual record=============//
	#pragma omp parallel for private(ir)
	for(ir=0; ir<r_n; ir++)
	{
		// ir = r_id[ix];
		// if(ir>=0)
		// {
		// p2[r_iz*ntx+r_ix[ir]] += record[it*r_n+ir] - record_obs[it*r_n+ir];
		p2[r_iz*ntx+r_ix[ir]] += record[it*r_n+ir]
						*dt*dt*velp[r_iz*ntx+r_ix[ir]]*velp[r_iz*ntx+r_ix[ir]];
		// }
	}

	//=======Calculate gradient=========//
	if(it%nt_interval==0)
	{
		#pragma omp parallel for collapse(2) private(iz, ix)
		for(ix=pml;ix<ntx-pml;ix++)
		{
			for(iz=pml;iz<ntz-pml;iz++)
			{
				grad[(iz-pml)*(ntx-2*pml)+(ix-pml)] += -2.0/powf(velp[iz*ntx+ix], 3) *
						psave[(int)(it/nt_interval)*ntx*ntz+iz*ntx+ix]*p2[iz*ntx+ix]*nt_interval*dt;
			}
		}
	}
}

void wavefield_IO(int forward_or_backward, int ntx, int ntz, int pml, int nt, int nt_interval, int it, 
				float dx, float dz, float dt, float *rik, float *velp, 
				float *p0, float *p1, float *p2, float *psave, float *record, float *grad,
				int s_ix, int s_iz, int r_iz, int *r_ix, int r_n)
{
	int ix, iz, ip, ir;

	//=======================Forward=======================//
	//=======================Forward=======================//
	if(forward_or_backward == 1)
	{
		//==============Add Ricker source=============//
		p2[s_iz*ntx+s_ix] += rik[it];

		//=======save forward wavefield to calculate gradient=========//
		if(it%nt_interval==0)
		{	
			#pragma omp parallel for collapse(2) private(iz, ix)
			for(ix=0;ix<ntx;ix++)
			{
				for(iz=0;iz<ntz;iz++)
				{
					psave[it/nt_interval*ntx*ntz+iz*ntx+ix] = 
							(p2[iz*ntx+ix]-2*p1[iz*ntx+ix]+p0[iz*ntx+ix])/dt/dt;
				}
			}
		}

		//==============Record seismograms=============//
		#pragma omp parallel for private(ir)
		for(ir=0; ir<r_n; ir++)
		{
			// ir = r_id[ix];
			// if(ir>=0)
			// {
			record[it*r_n+ir] = p2[r_iz*ntx+r_ix[ir]];
			// }
		}
	}

	//=======================Backward=======================//
	//=======================Backward=======================//
	if(forward_or_backward == 2)
	{
		//==============Back propagate the residual record=============//
		#pragma omp parallel for private(ir)
		for(ir=0; ir<r_n; ir++)
		{
			// ir = r_id[ix];
			// if(ir>=0)
			// {
			// p2[r_iz*ntx+r_ix[ir]] += record[it*r_n+ir] - record_obs[it*r_n+ir];
			p2[r_iz*ntx+r_ix[ir]] += record[it*r_n+ir];
			// }
		}

		//=======Calculate gradient=========//
		if(it%nt_interval==0)
		{
			#pragma omp parallel for collapse(2) private(iz, ix)
			for(ix=pml;ix<ntx-pml;ix++)
			{
				for(iz=pml;iz<ntz-pml;iz++)
				{
					grad[(iz-pml)*(ntx-2*pml)+(ix-pml)] += -2.0/powf(velp[iz*ntx+ix], 3) *
							psave[it/nt_interval*ntx*ntz+iz*ntx+ix]*p2[iz*ntx+ix]*nt_interval*dt;
				}
			}
		}
	}
}


void updata_p(int ntx, int ntz, float *p0, float *p1, float *p2)
{
	int ix, iz;

	#pragma omp parallel for collapse(2) private(iz,ix)
	for(ix=0;ix<ntx; ix++)
	{
		for(iz=0;iz<ntz;iz++)
		{
			p0[iz*ntx+ix] = p1[iz*ntx+ix];
			p1[iz*ntx+ix] = p2[iz*ntx+ix];
		}
	}
}


void wavefield_initialization(int ntx, int ntz, float *p0, float *p1, float *p2)
{
	int ix, iz;

	#pragma omp parallel for collapse(2) private(iz,ix)
	for(ix=0; ix<ntx; ix++)
		for(iz=0; iz<ntz; iz++)
		{
			p0[iz*ntx+ix] = 0.0;
			p1[iz*ntx+ix] = 0.0;
			p2[iz*ntx+ix] = 0.0;
		}
}


void getrik(int nt, float dt, float f0, float *rik)
{
	int it;	float tmp;

	// #pragma omp parallel for private(it, tmp) 
	for(it=0;it<nt;it++)
	{
		tmp=pow(pi*f0*(it*dt-1/f0),2.0);
		rik[it]=(float)((1.0-2.0*tmp)*exp(-tmp));		
	}

/*	
	float *a;
	float max=0.0;
	a=(float *) malloc(sizeof(float)*nt);
	
	for(it=1;it<=nt;it++)
	{
		tmp=pow(pi*f0*(it*dt-1.0/f0),2.0);
		a[it-1]=(float)((1.0-2.0*tmp)*exp(-tmp));		
	}
	
	for(it=1;it<nt;it++)
	{
		rik[it]=(a[it]-a[it-1])/dt;
		if(fabs(max)<fabs(rik[it]))
		{
			max=fabs(rik[it]);
		}
	}
	
	for(it=0;it<nt;it++)
	{
		rik[it]=rik[it]/max;
	}	
	
	FILE *fprik=fopen("rik.bin","wb");
	fwrite(rik,sizeof(float),nt,fprik);
	fclose(fprik);	
	free(a);
	*/
}


void get_velp(int pml, int ntx, int ntz, float *vel_inner, float *velp)
{
	int ix, iz,ip,ip0;

	// FILE *fp1=fopen(filevelp,"rb");
	// for(ix=pml;ix<ntx-pml;ix++)
	// 	for(iz=pml;iz<ntz-pml;iz++)
	// 			fread(&velp[iz*ntx+ix],sizeof(float),1,fp1);
	// fclose(fp1);

    #pragma omp parallel for collapse(2) private(iz,ix)
    for(ix=pml;ix<ntx-pml;ix++)
    {
		for(iz=pml;iz<ntz-pml;iz++)
		{
			velp[iz*ntx+ix]=vel_inner[(iz-pml)*(ntx-2*pml)+(ix-pml)];                      
		}  //inner
	}    

	#pragma omp parallel for collapse(2) private(iz,ix)
    for(ix=pml;ix<ntx-pml;ix++)
    {
		for(iz=0;iz<pml;iz++)
		{
			velp[iz*ntx+ix]=velp[pml*ntx+ix];                      
		}  //top
	}
	#pragma omp parallel for collapse(2) private(iz,ix)
    for(ix=pml;ix<ntx-pml;ix++)
    {	
		for(iz=ntz-pml;iz<ntz;iz++)
		{
			velp[iz*ntx+ix]=velp[(ntz-pml-1)*ntx+ix];               
		}
    }  //bottom
    
	#pragma omp parallel for collapse(2) private(iz,ix)	
    for(iz=0;iz<ntz;iz++)
    {
        for(ix=0;ix<pml;ix++)
        {
			velp[iz*ntx+ix]=velp[iz*ntx+pml];                 
		}	//left
	}
	#pragma omp parallel for collapse(2) private(iz,ix)
    for(iz=0;iz<ntz;iz++)
    {	
		for(ix=ntx-pml;ix<ntx;ix++)
		{
			velp[iz*ntx+ix]=velp[iz*ntx+(ntx-pml-1)];               
        }
    }

	// #pragma omp parallel for collapse(2) private(iz,ix)
	// for(ix=0;ix<ntx;ix++)
	// {
	// 	for(iz=0;iz<ntz;iz++)
	// 	{
	// 		velp[iz*ntx+ix]=2000.0;
	// 	}
	// }
	// #pragma omp parallel for collapse(2) private(iz,ix)
	// for(ix=0;ix<ntx;ix++)
	// {
	// 	for(iz=ntz/2;iz<ntz;iz++)
	// 	{
	// 		velp[iz*ntx+ix]=2000.0;
	// 	}					
	// }
/*    
	fp1=fopen("./input/velp1.bin","wb");
	for(ix=0;ix<ntx;ix++)
		for(iz=0;iz<ntz;iz++)
				fwrite(&velp[iz*ntx+ix],sizeof(float),1,fp1);
	fclose(fp1);
	
	fp1=fopen("./input/Qp1.bin","wb");
	for(ix=0;ix<ntx;ix++)
		for(iz=0;iz<ntz;iz++)
				fwrite(&Qp[iz*ntx+ix],sizeof(float),1,fp1);
	fclose(fp1);    
*/
}


void fd_coefficient(int Lc, float *fd)
{
	int m, i;
	float s1, s2;
	// for(m=1;m<=Lc;m++)
	// {
	// 	s1=1.0;s2=1.0;
	// 	for(i=1;i<m;i++)
	// 	{
	// 		s1=s1*(2.0*i-1)*(2.0*i-1);
	// 		s2=s2*((2.0*m-1)*(2.0*m-1)-(2.0*i-1)*(2.0*i-1));
	// 	}
	// 	for(i=m+1;i<=Lc;i++)
	// 	{
	// 		s1=s1*(2.0*i-1)*(2.0*i-1);
	// 		s2=s2*((2.0*m-1)*(2.0*m-1)-(2.0*i-1)*(2.0*i-1));
	// 	}
	// 	s2=fabs(s2);
	// 	fd[m-1]=pow(-1.0,m+1)*s1/(s2*(2.0*m-1));
	// }
	fd[3] = 1.0/90;
	fd[2] = -3.0/20;
	fd[1] = 1.5;
	fd[0] = -49.0/18;
}

void read_parameters(char inputfile[200], int *nx, int *nz, int *pml0, int *Lc, int *ns, int *nt, int *ds, 
						int *ns0, int *depths, int *depthr, int *dr, int *nt_interval,
						float *dx, float *dz, float *dt, float *f0)
{
	char strtmp[256];
	FILE *fp=fopen(inputfile,"r");
	if(fp==0){printf("Cannot open %s!\n", inputfile);exit(0);}

	read_int_value(strtmp, fp, nx);
	read_int_value(strtmp, fp, nz);
	read_int_value(strtmp, fp, pml0);
	read_int_value(strtmp, fp, Lc);
	read_int_value(strtmp, fp, ns);
	read_int_value(strtmp, fp, nt);
	read_int_value(strtmp, fp, ds);
	read_int_value(strtmp, fp, ns0);
	read_int_value(strtmp, fp, depths);
	read_int_value(strtmp, fp, depthr);
	read_int_value(strtmp, fp, dr);
	read_int_value(strtmp, fp, nt_interval);

	read_float_value(strtmp, fp, dx);
	read_float_value(strtmp, fp, dz);
	read_float_value(strtmp, fp, dt);
	read_float_value(strtmp, fp, f0);

	fclose(fp);

	return;
}

void read_int_value(char strtmp[256], FILE *fp, int *param)
{
	fgets(strtmp,256,fp);
	fscanf(fp,"\n");
	fscanf(fp,"%d",param);
	fscanf(fp,"\n");
	return;
}

void read_float_value(char strtmp[256], FILE *fp, float *param)
{
	fgets(strtmp,256,fp);
	fscanf(fp,"\n");
	fscanf(fp,"%f",param);
	fscanf(fp,"\n");
	return;
}
