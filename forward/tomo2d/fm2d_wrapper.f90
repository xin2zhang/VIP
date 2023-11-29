! wrapper to c
module fm2d_wrapper

    use iso_c_binding, only : c_double, c_int
    use omp_lib

    implicit none

contains

    subroutine fm2d(nsrc,srcx,srcy, &
            nrec, recx, recy, mask, &
            nx, ny, xmin, ymin, dx, dy, &
            gdx, gdy, sdx, sext, vel, time, dtdv,earth)
        use m_fm2d, only : modrays, T_RAY
        integer(c_int), intent(in) :: nsrc
        real(kind=c_double), dimension(:), intent(in) :: srcx
        real(kind=c_double), dimension(:), intent(in) :: srcy
        integer(c_int), intent(in) :: nrec
        real(kind=c_double), dimension(:), intent(in) :: recx
        real(kind=c_double), dimension(:), intent(in) :: recy
        integer(c_int), dimension(:,:), intent(in) :: mask
        integer(c_int), intent(in) :: nx, ny
        real(kind=c_double), intent(in) :: xmin, ymin
        real(kind=c_double), intent(in) :: dx, dy
        integer(c_int), intent(in) :: gdx, gdy
        integer(c_int), intent(in) :: sdx, sext 
        real(kind=c_double), dimension(:,:), intent(in) :: vel
        real(kind=c_double), dimension(:,:), intent(inout) :: time
        real(kind=c_double), dimension(:,:,:,:), intent(out) :: dtdv
        real(kind=c_double), intent(in) :: earth

        !real(kind=c_double), parameter :: earth = 6371
        integer :: crazyray
        type(T_RAY), dimension(:),allocatable :: rays
        real(kind=c_double) :: band

        !write(*,*) srcx
        !write(*,*) srcy
        !write(*,*) xmin,ymin
        !write(*,*) nx,ny
        !write(*,*) vel
        allocate(rays(nsrc*nrec))
        crazyray = 0
        band = 0.8
        time = 0.0
        dtdv = 0
        call modrays(nsrc,srcx,srcy,nrec,recx,recy, &
            mask,0,nx,ny,xmin,ymin,dx,dy,vel,&
            gdx,gdy,1,sdx,sext,earth, 1, band, time, &
            rays,dtdv,crazyray,0)

    end subroutine

    subroutine c_fm2d(nsrc,srcx,srcy,nrec,recx,recy,nx,ny,mask,&
            xmin,ymin,dx,dy,gdx,gdy,sdx, sext, vel,time,dtdv,earth) bind(c)
        integer(c_int), intent(in) :: nsrc
        real(kind=c_double), intent(in) :: srcx(nsrc)
        real(kind=c_double), intent(in) :: srcy(nsrc)
        integer(c_int), intent(in) :: nrec
        real(kind=c_double), intent(in) :: recx(nrec)
        real(kind=c_double), intent(in) :: recy(nrec)
        integer(c_int), intent(in) :: mask(nrec*nsrc,2)
        integer(c_int), intent(in) :: nx, ny
        real(kind=c_double), intent(in) :: xmin, ymin
        real(kind=c_double), intent(in) :: dx, dy
        integer(c_int), intent(in) :: gdx, gdy
        integer(c_int), intent(in) :: sdx, sext
        real(kind=c_double), intent(in) :: vel(ny,nx)
        real(kind=c_double), intent(out) :: time(nrec,nsrc)
        real(kind=c_double), intent(out) :: dtdv(ny,nx,nrec,nsrc)
        real(kind=c_double), intent(in) :: earth

        call fm2d(nsrc,srcx,srcy,nrec,recx,recy,mask,&
            nx,ny,xmin,ymin,dx,dy,gdx,gdy,sdx, sext, vel,&
            time,dtdv,earth)

    end subroutine

    subroutine c_fm2d_parallel(nsrc,srcx,srcy,nrec,recx,recy,nx,ny,mask,&
            xmin,ymin,dx,dy,gdx,gdy,sdx, sext,nv,vel,times,earth) bind(c)
        integer(c_int), intent(in) :: nsrc
        real(kind=c_double), intent(in) :: srcx(nsrc)
        real(kind=c_double), intent(in) :: srcy(nsrc)
        integer(c_int), intent(in) :: nrec
        real(kind=c_double), intent(in) :: recx(nrec)
        real(kind=c_double), intent(in) :: recy(nrec)
        integer(c_int), intent(in) :: mask(nrec*nsrc,2)
        integer(c_int), intent(in) :: nx, ny
        real(kind=c_double), intent(in) :: xmin, ymin
        real(kind=c_double), intent(in) :: dx, dy
        integer(c_int), intent(in) :: gdx, gdy
        integer(c_int), intent(in) :: sdx, sext
        integer(c_int), intent(in) :: nv
        real(kind=c_double), intent(in) :: vel(ny,nx,nv)
        real(kind=c_double), intent(out) :: times(nrec*nsrc,nv)
        real(kind=c_double), intent(in) :: earth

        real(kind=c_double)  :: time(nrec,nsrc)
        real(kind=c_double), dimension(:,:,:,:), allocatable  :: dtdv
        integer i, j
        
        allocate(dtdv(ny,nx,nrec,nsrc))
        !$omp parallel
        !$omp do private(time,dtdv,i,j)
        do i = 1, nv
            call fm2d(nsrc,srcx,srcy,nrec,recx,recy,mask,&
                nx,ny,xmin,ymin,dx,dy,gdx,gdy,sdx, sext, vel(:,:,i),&
                time,dtdv,earth)
            times(:,i)= reshape(time,(/nrec*nsrc/))
        enddo
        !$omp end do
        !$omp end parallel

    end subroutine

    subroutine c_fm2d_lglike(nsrc,srcx,srcy,nrec,recx,recy,nx,ny,mask,&
            xmin,ymin,dx,dy,gdx,gdy,sdx, sext,nv,vel,tobs,lglike,grads,earth) bind(c)
        integer(c_int), intent(in) :: nsrc
        real(kind=c_double), intent(in) :: srcx(nsrc)
        real(kind=c_double), intent(in) :: srcy(nsrc)
        integer(c_int), intent(in) :: nrec
        real(kind=c_double), intent(in) :: recx(nrec)
        real(kind=c_double), intent(in) :: recy(nrec)
        integer(c_int), intent(in) :: mask(nrec*nsrc,2)
        integer(c_int), intent(in) :: nx, ny
        real(kind=c_double), intent(in) :: xmin, ymin
        real(kind=c_double), intent(in) :: dx, dy
        integer(c_int), intent(in) :: gdx, gdy
        integer(c_int), intent(in) :: sdx, sext
        integer(c_int), intent(in) :: nv
        real(kind=c_double), intent(in) :: vel(ny,nx,nv)
        real(kind=c_double), intent(in) :: tobs(2,nrec*nsrc)
        real(kind=c_double), intent(out) :: lglike(nv)
        real(kind=c_double), intent(out) :: grads(ny*nx,nv)
        real(kind=c_double), intent(in) :: earth

        real(kind=c_double)  :: time(nrec,nsrc), time1d(nrec*nsrc)
        real(kind=c_double), dimension(:,:,:,:), allocatable  :: dtdv
        real(kind=c_double), dimension(:,:), allocatable  :: dtdv2d
        integer i, j
        
        allocate(dtdv(ny,nx,nrec,nsrc))
        allocate(dtdv2d(ny*nx,nrec*nsrc))
        lglike = 0.
        grads = 0
        !$omp parallel
        !$omp do private(time,time1d,dtdv,dtdv2d,i,j)
        do i = 1, nv
            call fm2d(nsrc,srcx,srcy,nrec,recx,recy,mask,&
                nx,ny,xmin,ymin,dx,dy,gdx,gdy,sdx, sext, vel(:,:,i),&
                time,dtdv,earth)
            time1d = reshape(time,(/nrec*nsrc/))
            dtdv2d = reshape(dtdv,(/ny*nx,nrec*nsrc/))
            lglike(i) = 0.5*sum((tobs(1,:)-time1d)**2/tobs(2,:)**2)
            do j = 1, nrec*nsrc
                grads(:,i) = grads(:,i) + dtdv2d(:,j)*(tobs(1,j)-time1d(j))/tobs(2,j)**2
            enddo
        enddo
        !$omp end do
        !$omp end parallel

    end subroutine

    subroutine c_many_fm2d(nsrc,srcx,srcy,nrec,recx,recy,nx,ny,&
            xmin,ymin,dx,dy,gdx,gdy,sdx,sext,nv,vel,tobs,mask,lglike,grads,earth) bind(c)
        integer(c_int), intent(in) :: nsrc
        real(kind=c_double), intent(in) :: srcx(nsrc)
        real(kind=c_double), intent(in) :: srcy(nsrc)
        integer(c_int), intent(in) :: nrec
        real(kind=c_double), intent(in) :: recx(nrec)
        real(kind=c_double), intent(in) :: recy(nrec)
        integer(c_int), intent(in) :: nx, ny
        real(kind=c_double), intent(in) :: xmin, ymin
        real(kind=c_double), intent(in) :: dx, dy
        integer(c_int), intent(in) :: gdx, gdy
        integer(c_int), intent(in) :: sdx, sext
        integer(c_int), intent(in) :: nv
        real(kind=c_double), intent(in) :: vel(ny,nx,nv)
        real(kind=c_double), intent(in) :: tobs(2,nrec*nsrc,nv)
        integer(c_int), intent(in) :: mask(2,nrec*nsrc,nv)
        real(kind=c_double), intent(out) :: lglike(nv)
        real(kind=c_double), intent(out) :: grads(ny*nx,nv)
        real(kind=c_double), intent(in) :: earth

        real(kind=c_double)  :: time(nrec,nsrc), time1d(nrec*nsrc)
        real(kind=c_double), dimension(:,:,:,:), allocatable  :: dtdv
        real(kind=c_double), dimension(:,:), allocatable  :: dtdv2d
        integer i, j
        
        allocate(dtdv(ny,nx,nrec,nsrc))
        allocate(dtdv2d(ny*nx,nrec*nsrc))
        lglike = 0.
        grads = 0
        !$omp parallel
        !$omp do private(time,time1d,dtdv,dtdv2d,i,j)
        do i = 1, nv
            call fm2d(nsrc,srcx,srcy,nrec,recx,recy,transpose(mask(:,:,i)),&
                nx,ny,xmin,ymin,dx,dy,gdx,gdy,sdx,sext,vel(:,:,i),&
                time,dtdv,earth)
            time1d = reshape(time,(/nrec*nsrc/))
            dtdv2d = reshape(dtdv,(/ny*nx,nrec*nsrc/))
            lglike(i) = 0.5*sum((tobs(1,:,i)-time1d)**2/tobs(2,:,i)**2)
            do j = 1, nrec*nsrc
                grads(:,i) = grads(:,i) + dtdv2d(:,j)*(tobs(1,j,i)-time1d(j))/tobs(2,j,i)**2
            enddo
        enddo
        !$omp end do
        !$omp end parallel

    end subroutine

end module
