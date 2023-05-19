!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
! MAIN PROGRAM
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
! TYPE: PROGRAM   
! CODE: FORTRAN 90
! This program is designed to implement the Fast Marching
! Method (FMM) for calculating first-arrival traveltimes
! through a 2-D continuous velocity medium in spherical shell
! coordinates (x=theta or latitude, z=phi or longitude). 
! It is written in Fortran 90, although it is probably more 
! accurately  described as Fortran 77 with some of the Fortran 90
! extensions.
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

! nru					! number of rays to update 
!Input is the same as if we were using the input file fm2dss.in
!nsrc,scx,scz				! sources (degrees)
!nrc,rcx,rcz				! receivers (degrees)
!srs					! source-receiver associations (as in first column of otimes.dat)
!nvx,nvz,goxd,gozd,dvxd,dvzd,velv	! velocity grid information (as in grid2d.vtx)
!gdx,gdz				! grid dicing in latitude and longitude
!asgr					! apply source grid refinement? (0=no,1=yes)
!sgdl,sgs				! dicing level and extent of refined grid
!earth					! Earth radius in km
!fom					! use first-order(0) or mixed-order(1) scheme
!snb					! narrow band size (0-1) as fraction of nnx*nnz
!ttime_all                              ! output, travel time for all the source receiver association
module m_fm2d

    use traveltime
    use globalp
    use omp_lib
    implicit none
    private

    public :: modrays
    public :: T_RAY

    real(kind=i10), parameter :: eps = 1E-8
    type T_RAY
        integer  srcid, revid
        integer  npoints
        real(kind=i10), dimension(:,:), allocatable :: points
     contains
        procedure :: length => ray_length
    end type


contains

function ray_length(this) result(length)
    class(T_RAY), intent(in) :: this
    real(kind=i10) :: length, dist
    integer i

    length = 0

    !if ray contains only one point, e.g. receiver point, exit
    if( this%npoints <= 1 .or. size(this%points,1) <= 1) return
    do i = 2, this%npoints
        dist = (this%points(1,i)-this%points(1,i-1))**2 + &
               (this%points(2,i)-this%points(2,i-1))**2
        length = length + sqrt(dist)
    enddo
end function ray_length

SUBROUTINE modrays(nsrcin,scxin,sczin,&
                nrcin,rcxin,rczin,&
                srsin,wrgfin,&
                nvxin,nvzin,goxdin,gozdin,dvxdin,dvzdin,velvin,&
                gdxin,gdzin,&
                asgrin,&
                sgdlin,sgsin,&
                earthin,&
                fomin,&
                snbin,&
                ttime,&
                rays, dtdv, crazyray,uar,&
                timeField,btime)

USE globalp
IMPLICIT NONE
real(kind=i10), dimension(:,:,:), intent(inout), optional :: timeField
real(kind=i10), dimension(:), intent(in), optional :: btime
INTEGER :: i,j,k,l,nsrc,tnr,urg,wrgf
INTEGER :: sgs,isx,isz,sw,idm1,idm2,nnxb,nnzb
INTEGER :: ogx,ogz,grdfx,grdfz,maxbt
REAL(KIND=i10) :: x,z,goxb,gozb,dnxb,dnzb
REAL(KIND=i10), DIMENSION (:), ALLOCATABLE :: scx,scz

INTEGER                 ::      nsrcin,nrcin,nvxin,nvzin,gdxin,gdzin,asgrin,sgdlin,sgsin,fomin,pn,wrgfin
INTEGER                 ::      srsin(nrcin*nsrcin,2)
INTEGER                 ::      nploci,ncloci

REAL(KIND=i10), intent(in)        ::      scxin(nsrcin),sczin(nsrcin),rcxin(nrcin),rczin(nrcin)
REAL(KIND=i10), intent(inout)     ::      ttime(nrcin,nsrcin)
real(kind=i10), dimension(:,:,:,:), intent(out)   :: dtdv
real(kind=i10), dimension(:,:,:), allocatable   :: fdm
type(T_RAY), dimension(:), intent(inout) :: rays
integer, intent(inout)                 ::      crazyray
REAL(KIND=i10)          ::      velvin(nvzin,nvxin)
REAL(KIND=i10)          ::      goxdin,gozdin
REAL(KIND=i10)          ::      dvxdin,dvzdin
REAL(KIND=i10)          ::      earthin,snbin
integer, intent(in)     ::      uar
integer                 ::      nru
logical :: dynamic


nru = sum(srsin(:,1),1)
!
! sources = File containing source locations
! receivers = File containing receiver locations
! grid = File containing grid of velocity vertices for
!        resampling on a finer grid with cubic B-splines
! frechet = output file containing matrix of frechet derivatives
! travelt = File name for storage of traveltime field
! wttf = Write traveltimes to file? (0=no,>0=source id)
! fom = Use first-order(0) or mixed-order(1) scheme
! nsrc = number of sources
! scx,scz = source location in r,x,z
! x,z = temporary variables for source location
! fsrt = find source-receiver traveltimes? (0=no,1=yes)
! rtravel = output file for source-receiver traveltimes
! cdum = dummy character variable
! wrgf = write ray geometries to file? (<0=all,0=no,>0=source id.)
! wrays = file containing raypath geometries
! cfd = calculate Frechet derivatives? (0=no, 1=yes)
! tnr = total numberof receivers
! sgs = Extent of refined source grid
! isx,isz = cell containing source
! nnxb,nnzb = Backup for nnz,nnx
! goxb,gozb = Backup for gox,goz
! dnxb,dnzb = Backup for dnx,dnz
! ogx,ogz = Location of refined grid origin
! gridfx,grdfz = Number of refined nodes per cell
! urg = use refined grid (0=no,1=yes,2=previously used)
! maxbt = maximum size of narrow band binary tree
! otimes = file containing source-receiver association information
!
wrgf=wrgfin
nsrc=nsrcin
ALLOCATE(scx(nsrc),scz(nsrc), STAT=checkstat)
scx=scxin
scz=sczin
nrc=nrcin
ALLOCATE(rcx(nrc),rcz(nrc), STAT=checkstat)
rcx=rcxin
rcz=rczin
if(earthin>0)then
   scx=(90.0-scx)*pi/180.0
   scz=scz*pi/180.0
   rcx=(90.0-rcx)*pi/180.0
   rcz=rcz*pi/180.0
endif
!!$omp parallel private(i,j,pn)&
!!$omp& private(x,z,isx,isz,sw,k,l,urg,nnxb,nnzb,dnxb,dnzb,goxb,gozb,idm1,idm2,maxbt,ogx,ogz,grdfx,grdfz)
ALLOCATE(srs(nrc,nsrc), STAT=checkstat)
ALLOCATE(srsv(nrc,nsrc), STAT=checkstat)
pn=0
DO i=1,nsrc
   DO j=1,nrc
        pn=pn+1
        srs(j,i)=srsin(pn,1)
        srsv(j,i)=srsin(pn,2)
   ENDDO
ENDDO
gdx=gdxin
gdz=gdzin
asgr=asgrin
sgdl=sgdlin
sgs=sgsin
fom=fomin
earth=earthin
snb=snbin

!
! Call a subroutine which reads in the velocity grid
!

CALL gridder(nvxin,nvzin,goxdin,gozdin,dvxdin,dvzdin,velvin)

! Now work out, source by source, the first-arrival traveltime
! field plus source-receiver traveltimes
! and ray paths if required. First, allocate memory to the
! traveltime field array
!
ALLOCATE(ttn(nnz,nnx), STAT=checkstat)
IF(checkstat > 0)THEN
   WRITE(*,*)'Error with ALLOCATE: PROGRAM fmm2dssRJu: REAL ttn'
ENDIF

IF(wrgf.NE.0)THEN
   !OPEN(UNIT=40,FILE=wrays,FORM='unformatted',STATUS='unknown')
   OPEN(UNIT=40,FILE='raysub.out',FORM='unformatted',STATUS='unknown')
   IF(wrgf.GT.0)THEN
      tnr=nrc
   ELSE
      tnr=nsrc*nrc
   ENDIF
   WRITE(40)tnr
   rbint=0
ENDIF

ALLOCATE(nsts(nnz,nnx))
maxbt=NINT(snb*nnx*nnz)
ALLOCATE(btg(maxbt))
!
! Loop through all sources and find traveltime fields
!
allocate(fdm(0:nvz+1,0:nvx+1,nrc))
fdm = 0
nploci=0
ncloci=0
!!$omp do
DO i=1,nsrc
    IF(SUM(srs(:,i)).EQ.0.AND.i.NE.1) THEN ! If there are no valid rays for source i (unless first source), then cycle the loop
        cycle
    endif
    x=scx(i)
    z=scz(i)
    
    isx=floor((x-gox)/dnx)+1
    isz=floor((z-goz)/dnz)+1
    sw=0
    IF(isx.lt.1.or.isx.gt.nnx)sw=1
    IF(isz.lt.1.or.isz.gt.nnz)sw=1
    IF(sw.eq.1)then
       !isx=90.0-x*180.0/pi
       !isz=z*180.0/pi
       WRITE(*,*)"Source lies outside bounds of model (lat,long)= ",90.0-x*180.0/pi, z*180.0/pi, gox, goz, dnx, dnz
       WRITE(*,*)"TERMINATING PROGRAM!!!"
       STOP
    ENDIF
    IF(isx.eq.nnx)isx=isx-1
    IF(isz.eq.nnz)isz=isz-1
   
    ! Now find rectangular box that extends outward from the nearest source node
    ! to "sgs" nodes away.
   
    vnl=isx-sgs
    IF(vnl.lt.1)vnl=1
    vnr=isx+sgs
    IF(vnr.gt.nnx)vnr=nnx
    vnt=isz-sgs
    IF(vnt.lt.1)vnt=1
    vnb=isz+sgs
    IF(vnb.gt.nnz)vnb=nnz

    ! If dynamic modelling, set nodes(<btime) to 0 and nodes(=btime) to 1 
    dynamic = .false.
    if(present(timeField) .and. present(btime))then
        if(btime(i)>maxval(timeField(vnt:vnb, vnl:vnr,i))) dynamic=.true.
    endif
    ! turn this off temporaliy, it seems that there is some instabilities
    dynamic = .false.

    if(dynamic)then
        ttn = timeField(:,:,i)
        where(ttn(:,:)<btime(i))
            nsts(:,:) = 0
        elsewhere(abs(ttn(:,:)-btime(i))<eps)
            nsts(:,:) = 1
        elsewhere
            nsts(:,:) = -1
        endwhere

        ! ensure the compelete of narrow band
        DO k=1,nnx 
           DO l=1,nnz
              IF(nsts(l,k).EQ.0)THEN
                 IF(l-1.GE.1)THEN
                    IF(nsts(l-1,k).EQ.-1)nsts(l,k)=1
                 ENDIF
                 IF(l+1.LE.nnz)THEN
                    IF(nsts(l+1,k).EQ.-1)nsts(l,k)=1
                 ENDIF
                 IF(k-1.GE.1)THEN
                    IF(nsts(l,k-1).EQ.-1)nsts(l,k)=1
                 ENDIF
                 IF(k+1.LE.nnx)THEN
                    IF(nsts(l,k+1).EQ.-1)nsts(l,k)=1
                 ENDIF
              ENDIF
           ENDDO
        ENDDO

        urg=2
        CALL travel(x,z,urg)
    else

        ! Begin by computing refined source grid if required
        urg=0
        IF(asgr.EQ.1)THEN
        
        !   Back up coarse velocity grid to a holding matrix
        
           IF(i.EQ.1)ALLOCATE(velnb(nnz,nnx))
           velnb=veln
           nnxb=nnx
           nnzb=nnz
           dnxb=dnx
           dnzb=dnz
           goxb=gox
           gozb=goz
        
        !   Identify nearest neighbouring node to source
        
           nrnx=(vnr-vnl)*sgdl+1
           nrnz=(vnb-vnt)*sgdl+1
           drnx=dvx/REAL(gdx*sgdl)
           drnz=dvz/REAL(gdz*sgdl)
           gorx=gox+dnx*(vnl-1)
           gorz=goz+dnz*(vnt-1)
           nnx=nrnx
           nnz=nrnz
           dnx=drnx
           dnz=drnz
           gox=gorx
           goz=gorz
        
        !   Reallocate velocity and traveltime arrays if nnx>nnxb or
        !   nnz>nnzb.
        
           IF(nnx.GT.nnxb.OR.nnz.GT.nnzb)THEN
              idm1=nnx
              IF(nnxb.GT.idm1)idm1=nnxb
              idm2=nnz
              IF(nnzb.GT.idm2)idm2=nnzb
              DEALLOCATE(veln,ttn,nsts,btg)
              ALLOCATE(veln(idm2,idm1))
              ALLOCATE(ttn(idm2,idm1))
              ALLOCATE(nsts(idm2,idm1))
              maxbt=NINT(snb*idm1*idm2)
              ALLOCATE(btg(maxbt))
           ENDIF
        
        !   Call a subroutine to compute values of refined velocity nodes
        
           CALL bsplrefine
        
        !   Compute first-arrival traveltime field through refined grid.
        
           urg=1
           CALL travel(x,z,urg)
        
        !   Now map refined grid onto coarse grid.
        
           ALLOCATE(ttnr(nnzb,nnxb))
           ALLOCATE(nstsr(nnzb,nnxb))
           IF(nnx.GT.nnxb.OR.nnz.GT.nnzb)THEN
              idm1=nnx
              IF(nnxb.GT.idm1)idm1=nnxb
              idm2=nnz
              IF(nnzb.GT.idm2)idm2=nnzb
              DEALLOCATE(ttnr,nstsr)
              ALLOCATE(ttnr(idm2,idm1))
              ALLOCATE(nstsr(idm2,idm1))
           ENDIF
           ttnr=ttn
           nstsr=nsts
           ogx=vnl
           ogz=vnt
           grdfx=sgdl
           grdfz=sgdl
           nsts=-1
           DO k=1,nnz,grdfz
              idm1=ogz+(k-1)/grdfz
              DO l=1,nnx,grdfx
                 idm2=ogx+(l-1)/grdfx
                 nsts(idm1,idm2)=nstsr(k,l)
                 IF(nsts(idm1,idm2).GE.0)THEN
                    ttn(idm1,idm2)=ttnr(k,l)
                 ENDIF
              ENDDO
           ENDDO
        
        !   Backup refined grid information
        
           nnxr=nnx
           nnzr=nnz
           goxr=gox
           gozr=goz
           dnxr=dnx
           dnzr=dnz
        
        !   Restore remaining values.
        
           nnx=nnxb
           nnz=nnzb
           dnx=dnxb
           dnz=dnzb
           gox=goxb
           goz=gozb
           DO j=1,nnx
              DO k=1,nnz 
                 veln(k,j)=velnb(k,j)
              ENDDO
           ENDDO
        
        !   Ensure that the narrow band is complete; if
        !   not, then some alive points will need to be
        !   made close.
        
           DO k=1,nnx ! why we need to cycle with all the grid ? note by xin.
              DO l=1,nnz
                 IF(nsts(l,k).EQ.0)THEN
                    IF(l-1.GE.1)THEN
                       IF(nsts(l-1,k).EQ.-1)nsts(l,k)=1
                    ENDIF
                    IF(l+1.LE.nnz)THEN
                       IF(nsts(l+1,k).EQ.-1)nsts(l,k)=1
                    ENDIF
                    IF(k-1.GE.1)THEN
                       IF(nsts(l,k-1).EQ.-1)nsts(l,k)=1
                    ENDIF
                    IF(k+1.LE.nnx)THEN
                       IF(nsts(l,k+1).EQ.-1)nsts(l,k)=1
                    ENDIF
                 ENDIF
              ENDDO
           ENDDO
        
        !   Finally, call routine for computing traveltimes once
        !   again.
        
           urg=2
           
           CALL travel(x,z,urg)
            
        ELSE
        
        !   Call a subroutine that works out the first-arrival traveltime
        !   field.
        
           CALL travel(x,z,urg)
        ENDIF
    endif
    
    if(present(timeField))then
        timeField(:,:,i) = ttn(1:nnz,1:nnx)
    endif

    ! Find source-receiver traveltimes if required
    call srtimes(x,z,i,ttime,nsrc)
     
    !Calculate raypath geometries and write to file if required.
    !Calculate Frechet derivatives with the same subroutine
    !if required.
    
    IF(uar .eq. 0 .OR. wrgf.eq.i.OR.wrgf.LT.0)THEN
        fdm = 0
        CALL rpaths(wrgf,uar,i,x,z,rays,fdm)
        dtdv(:,:,:,i) = fdm(0:nvz-1,0:nvx-1,:)
        crazyray=crazyray+crazyrp
    ENDIF
     
    IF(asgr.EQ.1 .and. allocated(ttnr)) DEALLOCATE(ttnr,nstsr)
ENDDO
!!$omp end do
!!$omp end parallel

IF(wrgf.NE.0)THEN

!  Notify about ray-boundary intersections if required.

   IF(rbint.EQ.1)THEN
      WRITE(*,*)'Note that at least one two-point ray path'
      WRITE(*,*)'tracked along the boundary of the model.'
      WRITE(*,*)'This class of path is unlikely to be'
      WRITE(*,*)'a true path, and it is STRONGLY RECOMMENDED'
      WRITE(*,*)'that you adjust the dimensions of your grid'
      WRITE(*,*)'to prevent this from occurring.'
   ENDIF
ENDIF
IF(asgr.EQ.1 .and. allocated(velnb))THEN
   DEALLOCATE (velnb, STAT=checkstat)
   IF(checkstat > 0)THEN
      WRITE(*,*)'Error with DEALLOCATE: PROGRAM fmm2dssRJu: velnb'
   ENDIF
ENDIF

DEALLOCATE (veln,ttn,scx,scz,nsts,btg,srs,srsv, STAT=checkstat)
IF(checkstat > 0)THEN
   WRITE(*,*)'Error with DEALLOCATE: PROGRAM fmm2dssRJu: final deallocate'
ENDIF
RETURN
END SUBROUTINE modrays

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
! TYPE: SUBROUTINE
! CODE: FORTRAN 90
! This subroutine is passed the name of the velocity
! grid file (grid) and reads in the velocity vertex values.
! The gridded values are globally shared via
! a MODULE statement. The values of the global propagation
! grid are also computed here.
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
SUBROUTINE gridder(nvxin,nvzin,goxdin,gozdin,dvxdin,dvzdin,velvin)
USE globalp
IMPLICIT NONE
INTEGER :: i,j,l,m,i1,j1,conx,conz,stx,stz
REAL(KIND=i10) :: u,sumi,sumj
REAL(KIND=i10), DIMENSION(:,:), ALLOCATABLE :: ui,vi

INTEGER                 ::      nvxin,nvzin
REAL(KIND=i10)          ::      goxdin,gozdin
REAL(KIND=i10)          ::      dvxdin,dvzdin
REAL(KIND=i10)          ::      velvin(nvzin,nvxin)

!
! u = independent parameter for b-spline
! ui,vi = bspline basis functions
! conx,conz = variables for edge of B-spline grid
! stx,stz = counters for veln grid points
! sumi,sumj = summation variables for computing b-spline

IF(ALLOCATED(velv)) DEALLOCATE(velv)
ALLOCATE(velv(0:nvzin+1,0:nvxin+1), STAT=checkstat)

!write(*,*)'Now in GRIDDER'

nvx=nvxin
nvz=nvzin
goxd=goxdin
gozd=gozdin
dvxd=dvxdin
dvzd=dvzdin
velv(1:nvzin,1:nvxin) = velvin
velv(0,:) = velv(1,:)
velv(nvzin+1,:) = velv(nvzin,:)
velv(:,0) = velv(:,1)
velv(:,nvxin+1) = velv(:,nvxin)

! Convert from degrees to radians
!
dvx=dvxd
dvz=dvzd
gox=goxd
goz=gozd
if(earth>0)then
   dvx=dvxd*pi/180.0
   dvz=dvzd*pi/180.0
   gox=(90.0-goxd)*pi/180.0
   goz=gozd*pi/180.0
endif
!
! Compute corresponding values for propagation grid.
!
nnx=(nvx-1)*gdx+1
nnz=(nvz-1)*gdz+1
dnx=dvx/gdx
dnz=dvz/gdz
dnxd=dvxd/gdx
dnzd=dvzd/gdz
ALLOCATE(veln(nnz,nnx), STAT=checkstat)
IF(checkstat > 0)THEN
   WRITE(*,*)'Error with ALLOCATE: SUBROUTINE gridder: REAL veln'
ENDIF
!
! Now dice up the grid
!
ALLOCATE(ui(gdx+1,4), STAT=checkstat)
IF(checkstat > 0)THEN
   WRITE(*,*)'Error with ALLOCATE: Subroutine gridder: REAL ui'
ENDIF
DO i=1,gdx+1
   u=gdx
   u=(i-1)/u
   ui(i,1)=(1.0-u)**3/6.0
   ui(i,2)=(4.0-6.0*u**2+3.0*u**3)/6.0
   ui(i,3)=(1.0+3.0*u+3.0*u**2-3.0*u**3)/6.0
   ui(i,4)=u**3/6.0
ENDDO
ALLOCATE(vi(gdz+1,4), STAT=checkstat)
IF(checkstat > 0)THEN
   WRITE(*,*)'Error with ALLOCATE: Subroutine gridder: REAL vi'
ENDIF
DO i=1,gdz+1
   u=gdz
   u=(i-1)/u
   vi(i,1)=(1.0-u)**3/6.0
   vi(i,2)=(4.0-6.0*u**2+3.0*u**3)/6.0
   vi(i,3)=(1.0+3.0*u+3.0*u**2-3.0*u**3)/6.0
   vi(i,4)=u**3/6.0
ENDDO
DO i=1,nvz-1
   conz=gdz
   IF(i==nvz-1)conz=gdz+1
   DO j=1,nvx-1
      conx=gdx
      IF(j==nvx-1)conx=gdx+1
      DO l=1,conz
         stz=gdz*(i-1)+l
         DO m=1,conx
            stx=gdx*(j-1)+m
            sumi=0.0
            DO i1=1,4
               sumj=0.0
               DO j1=1,4
                  sumj=sumj+ui(m,j1)*velv(i-2+i1,j-2+j1)
               ENDDO
               sumi=sumi+vi(l,i1)*sumj
            ENDDO
            veln(stz,stx)=sumi
         ENDDO
      ENDDO
   ENDDO
ENDDO
DEALLOCATE(ui,vi, STAT=checkstat)
IF(checkstat > 0)THEN
   WRITE(*,*)'Error with DEALLOCATE: SUBROUTINE gridder: REAL ui,vi'
ENDIF
END SUBROUTINE gridder

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
! TYPE: SUBROUTINE
! CODE: FORTRAN 90
! This subroutine is similar to bsplreg except that it has been
! modified to deal with source grid refinement
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
SUBROUTINE bsplrefine
USE globalp
INTEGER :: i,j,k,l,i1,j1,st1,st2,nrzr,nrxr
INTEGER :: origx,origz,conx,conz,idm1,idm2
REAL(KIND=i10) :: u,v
REAL(KIND=i10), DIMENSION (4) :: sum
REAL(KIND=i10), DIMENSION(gdx*sgdl+1,gdz*sgdl+1,4) :: ui,vi
!
! nrxr,nrzr = grid refinement level for source grid in x,z
! origx,origz = local origin of refined source grid
!
! Begin by calculating the values of the basis functions
!
nrxr=gdx*sgdl
nrzr=gdz*sgdl
DO i=1,nrzr+1
   v=nrzr
   v=(i-1)/v
   DO j=1,nrxr+1
      u=nrxr
      u=(j-1)/u
      ui(j,i,1)=(1.0-u)**3/6.0
      ui(j,i,2)=(4.0-6.0*u**2+3.0*u**3)/6.0
      ui(j,i,3)=(1.0+3.0*u+3.0*u**2-3.0*u**3)/6.0
      ui(j,i,4)=u**3/6.0
      vi(j,i,1)=(1.0-v)**3/6.0
      vi(j,i,2)=(4.0-6.0*v**2+3.0*v**3)/6.0
      vi(j,i,3)=(1.0+3.0*v+3.0*v**2-3.0*v**3)/6.0
      vi(j,i,4)=v**3/6.0
   ENDDO
ENDDO
!
! Calculate the velocity values.
!
origx=(vnl-1)*sgdl+1
origz=(vnt-1)*sgdl+1
DO i=1,nvz-1
   conz=nrzr
   IF(i==nvz-1)conz=nrzr+1
   DO j=1,nvx-1
      conx=nrxr
      IF(j==nvx-1)conx=nrxr+1
      DO k=1,conz
         st1=gdz*(i-1)+(k-1)/sgdl+1
         IF(st1.LT.vnt.OR.st1.GT.vnb)CYCLE
         st1=nrzr*(i-1)+k
         DO l=1,conx
            st2=gdx*(j-1)+(l-1)/sgdl+1
            IF(st2.LT.vnl.OR.st2.GT.vnr)CYCLE
            st2=nrxr*(j-1)+l
            DO i1=1,4
               sum(i1)=0.0
               DO j1=1,4
                  sum(i1)=sum(i1)+ui(l,k,j1)*velv(i-2+i1,j-2+j1)
               ENDDO
               sum(i1)=vi(l,k,i1)*sum(i1)
            ENDDO
            idm1=st1-origz+1
            idm2=st2-origx+1
            IF(idm1.LT.1.OR.idm1.GT.nnz)CYCLE
            IF(idm2.LT.1.OR.idm2.GT.nnx)CYCLE
            veln(idm1,idm2)=sum(1)+sum(2)+sum(3)+sum(4)
         ENDDO
      ENDDO
   ENDDO
ENDDO
END SUBROUTINE bsplrefine

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
! TYPE: SUBROUTINE
! CODE: FORTRAN 90
! This subroutine calculates all receiver traveltimes for
! a given source and writes the results to file.
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
SUBROUTINE srtimes(scx,scz,csid,ttime,nsrc)
USE globalp
!USE fm2dout
IMPLICIT NONE
INTEGER :: i,k,l,irx,irz,sw,isx,isz,csid,nsrc
!INTEGER, PARAMETER :: noray=0,yesray=1
REAL(KIND=i10) :: trr
REAL(KIND=i10), DIMENSION (nrc,nsrc) :: ttime
REAL(KIND=i10), PARAMETER :: norayt=0.0
REAL(KIND=i10) :: drx,drz,produ,scx,scz
REAL(KIND=i10) :: sred,dpl,rd1,vels,velr
REAL(KIND=i10), DIMENSION (2,2) :: vss
!
! irx,irz = Coordinates of cell containing receiver
! trr = traveltime value at receiver
! produ = dummy multiplier
! drx,drz = receiver distance from (i,j,k) grid node
! scx,scz = source coordinates
! isx,isz = source cell location
! sred = Distance from source to receiver
! dpl = Minimum path length in source neighbourhood.
! vels,velr = velocity at source and receiver
! vss = velocity at four grid points about source or receiver.
! csid = current source ID
! noray = switch to indicate no ray present
! norayt = default value given to null ray
! yesray = switch to indicate that ray is present
!
! Determine source-receiver traveltimes one at a time.
!
DO i=1,nrc
   IF(srs(i,csid).EQ.0)THEN
      ttime(i,csid) = norayt
      !WRITE(*,*)noray,norayt
      CYCLE
   ENDIF
! 
!  The first step is to locate the receiver in the grid.
!
   irx=floor((rcx(i)-gox)/dnx)+1
   irz=floor((rcz(i)-goz)/dnz)+1
   sw=0
   IF(irx.lt.1.or.irx.gt.nnx)sw=1
   IF(irz.lt.1.or.irz.gt.nnz)sw=1
   IF(sw.eq.1)then
      !irx=90.0-irx*180.0/pi
      !irz=irz*180.0/pi
      WRITE(*,*)"Receiver lies outside model (lat,long)= ",rcx(i), rcz(i)
      write(*,*) irx, irz
      write(*,*) nnz, nnx 
      WRITE(*,*)"TERMINATING PROGRAM!!!!"
      STOP
   ENDIF
   IF(irx.eq.nnx)irx=irx-1
   IF(irz.eq.nnz)irz=irz-1
!
!  Location of receiver successfully found within the grid. Now approximate
!  traveltime at receiver using bilinear interpolation from four
!  surrounding grid points. Note that bilinear interpolation is a poor
!  approximation when traveltime gradient varies significantly across a cell,
!  particularly near the source. Thus, we use an improved approximation in this
!  case. First, locate current source cell.
!
   isx=floor((scx-gox)/dnx)+1
   isz=floor((scz-goz)/dnz)+1
   dpl=dnx
   rd1=dnz
   if(earth>0)then
      dpl=dnx*earth
      rd1=dnz*earth*SIN(gox)
   endif
   IF(rd1.LT.dpl)dpl=rd1
   if(earth>0) rd1=dnz*earth*SIN(gox+(nnx-1)*dnx)
   IF(rd1.LT.dpl)dpl=rd1
   sred=(scx-rcx(i))**2
   sred=sred+(scz-rcz(i))**2
   if(earth>0)then
      sred=((scx-rcx(i))*earth)**2
      sred=sred+((scz-rcz(i))*earth*SIN(rcx(i)))**2
   endif
   sred=SQRT(sred)
   IF(sred.LT.dpl)sw=1
   IF(isx.EQ.irx)THEN
      IF(isz.EQ.irz)sw=1
   ENDIF
   IF(sw.EQ.1)THEN
!
!     Compute velocity at source and receiver
!
      DO k=1,2
         DO l=1,2
            vss(k,l)=veln(isz-1+l,isx-1+k)
         ENDDO
      ENDDO
      drx=(scx-gox)-(isx-1)*dnx
      drz=(scz-goz)-(isz-1)*dnz
      CALL bilinear(vss,drx,drz,vels)
      DO k=1,2
         DO l=1,2
            vss(k,l)=veln(irz-1+l,irx-1+k)
         ENDDO
      ENDDO
      drx=(rcx(i)-gox)-(irx-1)*dnx
      drz=(rcz(i)-goz)-(irz-1)*dnz
      CALL bilinear(vss,drx,drz,velr)
      trr=2.0*sred/(vels+velr)
   ELSE
      drx=(rcx(i)-gox)-(irx-1)*dnx
      drz=(rcz(i)-goz)-(irz-1)*dnz
      trr=0.0
      DO k=1,2
         DO l=1,2
            produ=(1.0-ABS(((l-1)*dnz-drz)/dnz))*(1.0-ABS(((k-1)*dnx-drx)/dnx))
            trr=trr+ttn(irz-1+l,irx-1+k)*produ
         ENDDO
      ENDDO
   ENDIF
   !ntt = ntt + 1
   ttime(i,csid) = trr
   !WRITE(*,*)yesray,trr
ENDDO
END SUBROUTINE srtimes



SUBROUTINE rpaths(wrgf,uar,csid,scx,scz,rays,fdm)
USE globalp
IMPLICIT NONE

type(T_RAY), dimension(:), intent(inout) :: rays
!INTEGER, PARAMETER :: i5=SELECTED_REAL_KIND(5,10)
INTEGER :: i,j,k,l,m,n,ipx,ipz,ipxr,ipzr,nrp,sw, jj
INTEGER :: wrgf,csid,ipxo,ipzo,isx,isz, cfd
integer :: uar
INTEGER :: ivx,ivz,ivxo,ivzo,nhp,maxrp
INTEGER :: ivxt,ivzt,ipxt,ipzt,igref
INTEGER :: ipxnew,ipznew,isxnew,isznew,ipxrnew,ipzrnew,swnew,nrpnew,igrefnew,jnew,ipxonew,ipzonew
INTEGER, DIMENSION (4) :: chp
REAL(KIND=i5), PARAMETER :: ftol=1.0e-6
REAL(KIND=i10) :: rayx,rayz	! rcxo and rczo added by EG
REAL(KIND=i10) :: dpl,rd1,rd2,xi,zi,vel,velo
REAL(KIND=i10) :: v,w,rigz,rigx,dinc,scx,scz
REAL(KIND=i10) :: dtx,dtz,drx,drz,produ,sred
REAL(KIND=i10) :: srednew
REAL(KIND=i10), DIMENSION (:), ALLOCATABLE :: rgx,rgz,rgxnew,rgznew
REAL(KIND=i10), DIMENSION (:,:,:) :: fdm
REAL(KIND=i10), DIMENSION (4) :: vrat,vi,wi,vio,wio

INTEGER totpts,srrecip

INTEGER, DIMENSION (:,:), ALLOCATABLE :: pnpts
REAL(KIND=i10), DIMENSION (:,:), ALLOCATABLE :: praypts
real(kind=i10) xdist, zdist
real(kind=i10) time_ray0, time_ray1

!write(*,*)'Hello from subroutine RPATHS'

IF(ALLOCATED(npts)) THEN
	DEALLOCATE(npts, STAT=checkstat)	! This stores the number of ray points from source csid to all receivers (valid paths only) that are passed to subroutine modrays
	IF(checkstat > 0)THEN
   		WRITE(*,*)'Error with DEALLOCATE: SUBROUTINE rpaths: npts'
	ENDIF
END IF
IF(ALLOCATED(raypts)) THEN
	DEALLOCATE(raypts, STAT=checkstat)	! This stores the ray points from source csid to all receivers (valid paths only) that are passed to subroutine modrays
	IF(checkstat > 0)THEN
   		WRITE(*,*)'Error with DEALLOCATE: SUBROUTINE rpaths: raypts'
	ENDIF
END IF
IF(ALLOCATED(pnpts)) THEN
	DEALLOCATE(pnpts)	! This array is temporary and only stores the number of ray points from source csid to one receiver
END IF
IF(ALLOCATED(praypts)) THEN
	DEALLOCATE(praypts)	! This array is temporary and only stores the ray points from source csid to one receiver
END IF

totpts=0
crazyrp=0
srrecip=0
cfd = 1
!
! ipx,ipz = Coordinates of cell containing current point
! ipxr,ipzr = Same as ipx,apz except for refined grid
! ipxo,ipzo = Coordinates of previous point
! rgx,rgz = (x,z) coordinates of ray geometry
! ivx,ivz = Coordinates of B-spline vertex containing current point
! ivxo,ivzo = Coordinates of previous point
! maxrp = maximum number of ray points
! nrp = number of points to describe ray
! dpl = incremental path length of ray
! xi,zi = edge of model coordinates
! dtx,dtz = components of gradT
! wrgf = Write out raypaths? (<0=all,0=no,>0=souce id)
! cfd = calculate Frechet derivatives? (0=no,1=yes)
! csid = current source id
! fdm = Frechet derivative matrix
! nhp = Number of ray segment-B-spline cell hit points
! vrat = length ratio of ray sub-segment
! chp = pointer to incremental change in x or z cell
! drx,drz = distance from reference node of cell
! produ = variable for trilinear interpolation
! vel = velocity at current point
! velo = velocity at previous point
! v,w = local variables of x,z
! vi,wi = B-spline basis functions at current point
! vio,wio = vi,wi for previous point
! ivxt,ivzt = temporary ivr,ivx,ivz values
! rigx,rigz = end point of sub-segment of ray path
! ipxt,ipzt = temporary ipx,ipz values
! dinc = path length of ray sub-segment
! rayr,rayx,rayz = ray path coordinates in single precision
! isx,isz = current source cell location
! scx,scz = current source coordinates
! sred = source to ray endpoint distance
! igref = ray endpoint lies in refined grid? (0=no,1=yes)
! nopath = switch to indicate that no path is present
!
! Allocate memory to arrays for storing ray path geometry
!
maxrp=nnx*nnz
ALLOCATE(rgx(maxrp+1), STAT=checkstat)
IF(checkstat > 0)THEN
   WRITE(*,*)'Error with ALLOCATE: SUBROUTINE rpaths: REAL rgx'
ENDIF
ALLOCATE(rgz(maxrp+1), STAT=checkstat)
IF(checkstat > 0)THEN
   WRITE(*,*)'Error with ALLOCATE: SUBROUTINE rpaths: REAL rgz'
ENDIF
!
! Allocate memory to partial derivative array
!
!IF(cfd.EQ.1)THEN
!   ALLOCATE(fdm(0:nvz+1,0:nvx+1), STAT=checkstat)
!   IF(checkstat > 0)THEN
!      WRITE(6,*)'Error with ALLOCATE: SUBROUTINE rpaths: REAL fdm'
!   ENDIF
!ENDIF
!
! Locate current source cell
!
IF(asgr.EQ.1)THEN
   isx=floor((scx-goxr)/dnxr)+1
   isz=floor((scz-gozr)/dnzr)+1
ELSE
   isx=floor((scx-gox)/dnx)+1
   isz=floor((scz-goz)/dnz)+1
ENDIF
!
! Set ray incremental path length equal to half width
! of cell
!
  dpl=dnx
  rd1=dnz
  if(earth>0)then
      dpl=dnx*earth
      rd1=dnz*earth*SIN(gox)
  endif
  IF(rd1.LT.dpl)dpl=rd1
  if(earth>0) rd1=dnz*earth*SIN(gox+(nnx-1)*dnx)
  IF(rd1.LT.dpl)dpl=rd1
  dpl=0.5*dpl
  !dpl = dpl
!
! Loop through all the receivers
!
DO i=1,nrc
!
!  If path does not exist, then cycle the loop
!
   IF(cfd.EQ.1)THEN
      fdm(:,:,i)=0.0
   ENDIF
   IF(srs(i,csid).EQ.0)THEN
      CYCLE
   ENDIF
!
!  The first step is to locate the receiver in the grid.
!
   ipx=floor((rcx(i)-gox)/dnx)+1
   ipz=floor((rcz(i)-goz)/dnz)+1
   sw=0
   IF(ipx.lt.1.or.ipx.ge.nnx)sw=1
   IF(ipz.lt.1.or.ipz.ge.nnz)sw=1
   IF(sw.eq.1)then
      !ipx=90.0-ipx*180.0/pi
      !ipz=ipz*180.0/pi
      WRITE(*,*)"Receiver lies outside model (lat,long)= ",rcx(i), rcz(i)
      WRITE(*,*)"TERMINATING PROGRAM!!!"
      STOP
   ENDIF
   IF(ipx.eq.nnx)ipx=ipx-1
   IF(ipz.eq.nnz)ipz=ipz-1
!
!  First point of the ray path is the receiver
!
   rgx(1)=rcx(i)
   rgz(1)=rcz(i)
   nrp = 1
!
!  Test to see if receiver is in source neighbourhood
!
   sred=(scx-rgx(1))**2
   sred=sred+((scz-rgz(1)))**2
   if(earth>0)then
       sred=((scx-rgx(1))*earth)**2
       sred=sred+((scz-rgz(1))*earth*SIN(rgx(1)))**2
   endif
   sred=SQRT(sred)
   IF(sred.LT.2.0*dpl)THEN
      rgx(2)=scx
      rgz(2)=scz
      nrp=2
      sw=1
   ENDIF
!
!  If required, see if receiver lies within refined grid
!
   IF(asgr.EQ.1)THEN
      ipxr=floor((rcx(i)-goxr)/dnxr)+1
      ipzr=floor((rcz(i)-gozr)/dnzr)+1
      igref=1
      IF(ipxr.LT.1.OR.ipxr.GE.nnxr)igref=0
      IF(ipzr.LT.1.OR.ipzr.GE.nnzr)igref=0
      IF(igref.EQ.1)THEN
         IF(nstsr(ipzr,ipxr).NE.0.OR.nstsr(ipzr+1,ipxr).NE.0)igref=0
         IF(nstsr(ipzr,ipxr+1).NE.0.OR.nstsr(ipzr+1,ipxr+1).NE.0)igref=0
      ENDIF
   ELSE
      igref=0
   ENDIF
!
!  Due to the method for calculating traveltime gradient, if the
!  the ray end point lies in the source cell, then we are also done.
!
   IF(sw.EQ.0)THEN
      IF(asgr.EQ.1)THEN
         IF(igref.EQ.1)THEN
            IF(ipxr.EQ.isx)THEN
               IF(ipzr.EQ.isz)THEN
                  rgx(2)=scx
                  rgz(2)=scz
                  nrp=2
                  sw=1
               ENDIF
            ENDIF
         ENDIF
      ELSE
         IF(ipx.EQ.isx)THEN
            IF(ipz.EQ.isz)THEN
               rgx(2)=scx
               rgz(2)=scz
               nrp=2
               sw=1
            ENDIF
         ENDIF
      ENDIF
   ENDIF
!
!  Now trace ray from receiver to "source"
!
   if(igref==0)then
       time_ray0 = ttn(ipz,ipx)
   else
       time_ray0 = ttnr(ipzr,ipxr)
   endif
   DO j=1,maxrp
      IF(sw.EQ.1)EXIT
!
!     Calculate traveltime gradient vector for current cell using
!     a first-order or second-order scheme.
!
      IF(igref.EQ.1)THEN
!
!        In this case, we are in the refined grid.
!
!        First order scheme applied here.
!
		   xdist=(rgx(j)-(goxr+dnxr*(ipxr-1)))/dnxr
	 	   zdist=(rgz(j)-(gozr+dnzr*(ipzr-1)))/dnzr

         if(ipxr == 1)then
             dtx = ttnr(ipzr,ipxr+1) - ttnr(ipzr,ipxr)
             dtx=dtx/dnxr
             if(earth>0) dtx=dtx/(earth*dnxr)
         elseif(ipxr ==  nnxr)then
             dtx = ttnr(ipzr,ipxr) - ttnr(ipzr,ipxr-1)
             dtx=dtx/dnxr
             if(earth>0) dtx=dtx/(earth*dnxr)
         else
             dtx=ttnr(ipzr,ipxr+1)-ttnr(ipzr,ipxr-1)
             !dtx=xdist*dtx+(1-xdist)*(ttnr(ipzr+1,ipxr+1)-ttnr(ipzr+1,ipxr-1))
             !dtx=dtx/(2.0*earth*dnxr)
             dtx=dtx/(2.0*dnxr)
             if(earth>0) dtx=dtx/(2*earth*dnxr)
         endif
         if(ipzr == 1)then
             dtz=ttnr(ipzr+1,ipxr)-ttnr(ipzr,ipxr)
             dtz = dtz/dnzr
             if(earth>0) dtz = dtz/(earth*SIN(rgx(j))*dnzr)
         elseif(ipzr == nnzr)then
             dtz=ttnr(ipzr,ipxr)-ttnr(ipzr-1,ipxr)
             dtz = dtz/dnzr
             if(earth>0) dtz = dtz/(earth*SIN(rgx(j))*dnzr)
         else
             dtz=ttnr(ipzr+1,ipxr)-ttnr(ipzr-1,ipxr)
             !dtz=zdist*dtz+(1-zdist)*(ttnr(ipzr+1,ipxr+1)-ttnr(ipzr-1,ipxr+1))
             !dtz=dtz/(2.0*earth*SIN(rgx(j))*dnzr)
             dtz=dtz/(2.0*dnzr)
             if(earth>0) dtz = dtz/(2*earth*SIN(rgx(j))*dnzr)
         endif
      ELSE
!
!        Here, we are in the coarse grid.
!
!        First order scheme applied here.
		   xdist=(rgx(j)-(gox+dnx*(ipx-1)))/dnx
	 	   zdist=(rgz(j)-(goz+dnz*(ipz-1)))/dnz
!
         if(ipx == 1)then
             dtx = ttn(ipz,ipx+1) - ttn(ipz,ipx)
             dtx=dtx/dnx
             if(earth>0) dtx=dtx/(earth*dnx)
         elseif(ipx ==  nnx)then
             dtx = ttn(ipz,ipx) - ttn(ipz,ipx-1)
             dtx=dtx/dnx
             if(earth>0) dtx=dtx/(earth*dnx)
         else
             dtx=ttn(ipz,ipx+1)-ttn(ipz,ipx-1)
             !dtx=xdist*dtx+(1-xdist)*(ttnr(ipzr+1,ipxr+1)-ttnr(ipzr+1,ipxr-1))
             !dtx=dtx/(2.0*earth*dnxr)
             dtx=dtx/(2.0*dnx)
             if(earth>0) dtx=dtx/(2*earth*dnx)
         endif
         if(ipz == 1)then
             dtz=ttn(ipz+1,ipx)-ttn(ipz,ipx)
             dtz = dtz/dnz
             if(earth>0) dtz = dtz/(earth*SIN(rgx(j))*dnz)
         elseif(ipzr == nnzr)then
             dtz=ttn(ipz,ipx)-ttn(ipz-1,ipx)
             dtz = dtz/dnz
             if(earth>0) dtz = dtz/(earth*SIN(rgx(j))*dnz)
         else
             dtz=ttn(ipz+1,ipx)-ttn(ipz-1,ipx)
             !dtz=zdist*dtz+(1-zdist)*(ttnr(ipzr+1,ipxr+1)-ttnr(ipzr-1,ipxr+1))
             !dtz=dtz/(2.0*earth*SIN(rgx(j))*dnzr)
             dtz=dtz/(2.0*dnz)
             if(earth>0) dtz = dtz/(2*earth*SIN(rgx(j))*dnz)
         endif
      ENDIF
      !if(csid==1.and.i==60)then
      !    write(*,*) igref, ttn(ipz+2,ipx), ttn(ipz+1,ipx), ttn(ipz,ipx), ttn(ipz-1,ipx), dtz
      !    rd1=SQRT(dtx**2+dtz**2)
      !    write(*,*) dtx, rd1
      !    write(*,*) rgx(j), rgz(j)
      !    write(*,*) rgx(j)-dpl*dtx/rd1, rgz(j)-dpl*dtz/rd1
      !endif
!
!     Calculate the next ray path point
!
      rd1=SQRT(dtx**2+dtz**2)
      rgx(j+1)=rgx(j)-dpl*dtx/rd1
      rgz(j+1)=rgz(j)-dpl*dtz/rd1
      if(earth>0)then
         rgx(j+1)=rgx(j)-dpl*dtx/(earth*rd1)
         rgz(j+1)=rgz(j)-dpl*dtz/(earth*SIN(rgx(j))*rd1)
      endif
!
!     Determine which cell the new ray endpoint
!     lies in.
!
      ipxo=ipx
      ipzo=ipz
      IF(asgr.EQ.1)THEN
!
!        Here, we test to see whether the ray endpoint lies
!        within a cell of the refined grid
!
         ipxr=floor((rgx(j+1)-goxr)/dnxr)+1
         ipzr=floor((rgz(j+1)-gozr)/dnzr)+1
         igref=1
         IF(ipxr.LT.1.OR.ipxr.GE.nnxr)igref=0
         IF(ipzr.LT.1.OR.ipzr.GE.nnzr)igref=0
         IF(igref.EQ.1)THEN
            IF(nstsr(ipzr,ipxr).NE.0.OR.nstsr(ipzr+1,ipxr).NE.0)igref=0
            IF(nstsr(ipzr,ipxr+1).NE.0.OR.nstsr(ipzr+1,ipxr+1).NE.0)igref=0
         ENDIF
         ipx=floor((rgx(j+1)-gox)/dnx)+1
         ipz=floor((rgz(j+1)-goz)/dnz)+1
      ELSE
         ipx=floor((rgx(j+1)-gox)/dnx)+1
         ipz=floor((rgz(j+1)-goz)/dnz)+1
         igref=0
      ENDIF
!
!     Test the proximity of the source to the ray end point.
!     If it is less than dpl then we are done
!
      sred=(scx-rgx(j+1))**2
      sred=sred+(scz-rgz(j+1))**2
      if(earth>0)then
         sred=((scx-rgx(j+1))*earth)**2
         sred=sred+((scz-rgz(j+1))*earth*SIN(rgx(j+1)))**2
      endif
      sred=SQRT(sred)
      sw=0      
      IF(sred.LT.2.0*dpl)THEN
      	  rgx(j+2)=scx
          rgz(j+2)=scz
          nrp=j+2
          sw=1
          IF(cfd.NE.1) EXIT
      ENDIF
!
!     Due to the method for calculating traveltime gradient, if the
!     the ray end point lies in the source cell, then we are also done.
!
      IF(sw.EQ.0)THEN
      	 IF(asgr.EQ.1)THEN
            IF(igref.EQ.1)THEN
               IF(ipxr.EQ.isx)THEN
                  IF(ipzr.EQ.isz)THEN
                     rgx(j+2)=scx
                     rgz(j+2)=scz
                     nrp=j+2
                     sw=1
                     IF(cfd.NE.1) EXIT
                  ENDIF
               ENDIF
            ENDIF
         ELSE
            IF(ipx.EQ.isx)THEN
               IF(ipz.EQ.isz)THEN	      
                  rgx(j+2)=scx
                  rgz(j+2)=scz
                  nrp=j+2
                  sw=1
                  IF(cfd.NE.1) EXIT
               ENDIF
            ENDIF
         ENDIF
      ENDIF
!
!     Test whether ray path segment extends beyond
!     box boundaries
!
      IF(ipx.LT.1)THEN
      	 rgx(j+1)=gox
         ipx=1
         rbint=1
      ENDIF
      IF(ipx.GE.nnx)THEN
         rgx(j+1)=gox+(nnx-1)*dnx
         ipx=nnx-1
         rbint=1
      ENDIF
      IF(ipz.LT.1)THEN
         rgz(j+1)=goz
         ipz=1
         rbint=1
      ENDIF
      IF(ipz.GE.nnz)THEN
         rgz(j+1)=goz+(nnz-1)*dnz
         ipz=nnz-1
         rbint=1
      ENDIF

      ! if the new point has larger travel time, something wrong has happened
      !if(igref==1)then
      !    time_ray1 = ttnr(ipzr,ipxr)
      !else
      !    time_ray1 = ttn(ipz,ipx)
      !endif
      !if((time_ray1-time_ray0)>1E-5)then
      !    time_ray1 = ttnr(ipzr,ipxr)
      !    write(*,*) 'Error with ray path'
      !    write(*,*) 'time: ', time_ray0, time_ray1
	  !    WRITE(*,*)'source id: ',csid
	  !    WRITE(*,*)'receiver id: ',i
	  !    WRITE(*,*)'grid: ', ipxo, ipzo, ipx, ipz, ipxr, ipzr
      !    write(*,*) igref, ttn(ipzo+2,ipxo), ttn(ipzo+1,ipxo), ttn(ipzo,ipxo), ttn(ipzo-1,ipxo), dtz
      !    write(*,*) igref, ttn(ipzo,ipxo+2), ttn(ipzo,ipxo+1), ttn(ipzo,ipxo), ttn(ipzo,ipxo-1), dtx
      !    write(*,*) rgx(j), rgz(j)
      !    write(*,*) rgx(j+1), rgz(j+1)
      !    stop
      !endif
      !time_ray0 = time_ray1


!      
!     If the maximum number of points maxrp has been reached, try and exchange source and receiver position and trace the ray again
!
      IF(j.EQ.maxrp-1.AND.sw.EQ.0)THEN
          WRITE(12,*)'WARNING: ERROR WITH RAYPATH!!!'
          WRITE(12,*)'Maximum number of points has been reached for ray with'
          WRITE(12,*)'source id: ',csid
          WRITE(12,*)'receiver id: ',i
          !WRITE(*,*)'on processor',procn
	
          crazyrp=crazyrp+1
          sw=1
          fdm(:,:,i) = 0
          EXIT
      ENDIF
!
!     Calculate the Frechet derivatives if required.
!
      IF(cfd.EQ.1)THEN
!
!        First determine which B-spline cell the refined cells
!        containing the ray path segment lies in. If they lie
!        in more than one, then we need to divide the problem
!        into separate parts (up to three).
!
         ivx=int((ipx-1)/gdx)+1
         ivz=int((ipz-1)/gdz)+1
         ivxo=int((ipxo-1)/gdx)+1
         ivzo=int((ipzo-1)/gdz)+1
!
!        Calculate up to two hit points between straight
!        ray segment and cell faces.
!
         nhp=0
         IF(ivx.NE.ivxo)THEN
            nhp=nhp+1
            IF(ivx.GT.ivxo)THEN
               xi=gox+(ivx-1)*dvx
            ELSE
               xi=gox+ivx*dvx
            ENDIF
            vrat(nhp)=(xi-rgx(j))/(rgx(j+1)-rgx(j))
            chp(nhp)=1
         ENDIF
         IF(ivz.NE.ivzo)THEN
            nhp=nhp+1
            IF(ivz.GT.ivzo)THEN
               zi=goz+(ivz-1)*dvz
            ELSE
               zi=goz+ivz*dvz
            ENDIF
            rd1=(zi-rgz(j))/(rgz(j+1)-rgz(j))
            IF(nhp.EQ.1)THEN
               vrat(nhp)=rd1
               chp(nhp)=2
            ELSE
               IF(rd1.GE.vrat(nhp-1))THEN
                  vrat(nhp)=rd1
                  chp(nhp)=2
               ELSE
                  vrat(nhp)=vrat(nhp-1)
                  chp(nhp)=chp(nhp-1)
                  vrat(nhp-1)=rd1
                  chp(nhp-1)=2
               ENDIF
            ENDIF
         ENDIF
         nhp=nhp+1
         vrat(nhp)=1.0
         chp(nhp)=0
!
!         Calculate the velocity, v and w values of the
!         first point
!
         drx=(rgx(j)-gox)-(ipxo-1)*dnx
         drz=(rgz(j)-goz)-(ipzo-1)*dnz
         vel=0.0
         DO l=1,2
            DO m=1,2
               produ=(1.0-ABS(((m-1)*dnz-drz)/dnz))
               produ=produ*(1.0-ABS(((l-1)*dnx-drx)/dnx))
               IF(ipzo-1+m.LE.nnz.AND.ipxo-1+l.LE.nnx)THEN
                  vel=vel+veln(ipzo-1+m,ipxo-1+l)*produ
               ENDIF
            ENDDO
         ENDDO
         drx=(rgx(j)-gox)-(ivxo-1)*dvx
         drz=(rgz(j)-goz)-(ivzo-1)*dvz
         v=drx/dvx
         w=drz/dvz
!
!        Calculate the 12 basis values at the point
!
         vi(1)=(1.0-v)**3/6.0
         vi(2)=(4.0-6.0*v**2+3.0*v**3)/6.0
         vi(3)=(1.0+3.0*v+3.0*v**2-3.0*v**3)/6.0
         vi(4)=v**3/6.0
         wi(1)=(1.0-w)**3/6.0
         wi(2)=(4.0-6.0*w**2+3.0*w**3)/6.0
         wi(3)=(1.0+3.0*w+3.0*w**2-3.0*w**3)/6.0
         wi(4)=w**3/6.0
         ivxt=ivxo
         ivzt=ivzo
!
!        Now loop through the one or more sub-segments of the
!        ray path segment and calculate partial derivatives
!
         DO k=1,nhp
            velo=vel
            vio=vi
            wio=wi
            IF(k.GT.1)THEN
               IF(chp(k-1).EQ.1)THEN
                  ivxt=ivx
               ELSE IF(chp(k-1).EQ.2)THEN
                  ivzt=ivz
               ENDIF
            ENDIF
!
!           Calculate the velocity, v and w values of the
!           new point
!
            rigz=rgz(j)+vrat(k)*(rgz(j+1)-rgz(j))
            rigx=rgx(j)+vrat(k)*(rgx(j+1)-rgx(j))
            ipxt=floor((rigx-gox)/dnx)+1
            ipzt=floor((rigz-goz)/dnz)+1
            drx=(rigx-gox)-(ipxt-1)*dnx
            drz=(rigz-goz)-(ipzt-1)*dnz
            vel=0.0
            DO m=1,2
               DO n=1,2
                  produ=(1.0-ABS(((n-1)*dnz-drz)/dnz))
                  produ=produ*(1.0-ABS(((m-1)*dnx-drx)/dnx))
                  IF(ipzt-1+n.LE.nnz.AND.ipxt-1+m.LE.nnx)THEN
                     vel=vel+veln(ipzt-1+n,ipxt-1+m)*produ
                  ENDIF
               ENDDO
            ENDDO
            drx=(rigx-gox)-(ivxt-1)*dvx
            drz=(rigz-goz)-(ivzt-1)*dvz
            v=drx/dvx
            w=drz/dvz
!
!           Calculate the 8 basis values at the new point
!
            vi(1)=(1.0-v)**3/6.0
            vi(2)=(4.0-6.0*v**2+3.0*v**3)/6.0
            vi(3)=(1.0+3.0*v+3.0*v**2-3.0*v**3)/6.0
            vi(4)=v**3/6.0
            wi(1)=(1.0-w)**3/6.0
            wi(2)=(4.0-6.0*w**2+3.0*w**3)/6.0
            wi(3)=(1.0+3.0*w+3.0*w**2-3.0*w**3)/6.0
            wi(4)=w**3/6.0
!
!           Calculate the incremental path length
!
            IF(k.EQ.1)THEN
               dinc=vrat(k)*dpl
            ELSE
               dinc=(vrat(k)-vrat(k-1))*dpl
            ENDIF
!
!           Now compute the 16 contributions to the partial
!           derivatives.
!
            DO l=1,4
               DO m=1,4
                  rd1=vi(m)*wi(l)/vel**2
                  rd2=vio(m)*wio(l)/velo**2
                  rd1=-(rd1+rd2)*dinc/2.0
                  rd2=fdm(ivzt-2+l,ivxt-2+m,i)
                  fdm(ivzt-2+l,ivxt-2+m,i)=rd1+rd2
               ENDDO
            ENDDO
         ENDDO
      ENDIF
      !IF(j.EQ.maxrp.AND.sw.EQ.0)THEN
      !	 WRITE(*,*)'Error with ray path detected!!! Processor',procn
      !   WRITE(*,*)'Source id: ',csid,'on processor',procn
      !   WRITE(*,*)'Receiver id: ',i,'on processor',procn
      !ENDIF
   ENDDO
!
!  Write ray paths to output file
!
   !IF(uar == 0 .or. wrgf.EQ.csid.OR.wrgf.LT.0)THEN
   !   if(.not. allocated(pnpts) )ALLOCATE(pnpts(1,2))
   !	  ALLOCATE(praypts(2,nrp))
   !   !WRITE(40)nrp
   !   totpts=totpts+nrp
   !   pnpts(1,1)=nrp
   !   pnpts(1,2)=srsv(i,csid)
   !   DO j=1,nrp
   !   	 rayx=rgx(j)
   !      rayz=rgz(j)
   !      praypts(1,j)=rayx
   !      praypts(2,j)=rayz
   !      !WRITE(40)rayx,rayz
   !   ENDDO
   !   
   !   !write(*,*) i, csid, srsv(i,csid), pnpts(1,2)
   !   !allocate( rays(pnpts(1,2))%points(2,nrp) )
   !   !rays(pnpts(1,2))%npoints = nrp
   !   !rays(pnpts(1,2))%points = praypts
   !   !rays(pnpts(1,2))%srcid = csid
   !   !rays(pnpts(1,2))%revid = i
   !   !deallocate( pnpts )
   !   deallocate( praypts )
   !ENDIF
   
!
!  Write partial derivatives to output file
!
   !IF(cfd.EQ.1)THEN
!
!     Determine the number of non-zero elements.
!
      !isum=0
      !DO j=0,nvz+1
      !   DO k=0,nvx+1
      !      IF(ABS(fdm(j,k)).GE.ftol)isum=isum+1
      !   ENDDO
      !ENDDO
      !WRITE(50)isum
      !isum=0
      !DO j=0,nvz+1
      !   DO k=0,nvx+1
      !      isum=isum+1
      !      IF(ABS(fdm(j,k)).GE.ftol)WRITE(50)isum,fdm(j,k)
      !   ENDDO
      !ENDDO
   !ENDIF
ENDDO
!IF(cfd.EQ.1)THEN
!   DEALLOCATE(fdm, STAT=checkstat)
!   IF(checkstat > 0)THEN
!      WRITE(6,*)'Error with DEALLOCATE: SUBROUTINE rpaths: fdm'
!   ENDIF
!ENDIF
1866	CONTINUE
IF(ALLOCATED(rgx)) DEALLOCATE(rgx,rgz, STAT=checkstat)
IF(checkstat > 0)THEN
   WRITE(*,*)'Error with DEALLOCATE: SUBROUTINE rpaths: rgx,rgz'
ENDIF
END SUBROUTINE rpaths
end module m_fm2d
