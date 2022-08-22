module m_kernel

    use omp_lib
    use iso_c_binding
    use m_utils, only : dp, sp, N_THREADS
    implicit none
    real(kind=dp), parameter :: eps = 1E-8

contains

    subroutine sqrd_c(m, n, x, w, dist) bind(c)
        implicit none
        integer(kind=c_int), intent(in), value :: m, n
        real(kind=dp), intent(in) :: x(n,m)
        real(kind=dp), intent(in) :: w(n)
        real(kind=dp), intent(out) :: dist(m,m)

        call sqr_dist(x,w,dist)

    end subroutine

    subroutine sqr_dist(x, w, dist)
        implicit none
        real(kind=dp), dimension(:,:), intent(in) :: x
        real(kind=dp), dimension(:), intent(in) :: w
        real(kind=dp), dimension(:,:), intent(out) :: dist

        integer i, j

        !omp parallel private(i,j) num_threads(N_THREADS)
        !$omp parallel private(i,j)
        !$omp do 
        do i = 1, size(x,2)
            do j = 1, size(x,2)
                dist(j,i) = sum((x(:,i)-x(:,j))**2 * w)
            enddo
        enddo
        !$omp end do
        !$omp end parallel
    end subroutine

    subroutine rbf_kernel(x, grad, dist, h, grad_out)
        implicit none
        real(kind=dp), dimension(:,:), intent(in) :: x
        real(kind=dp), dimension(:,:), intent(in) :: grad
        real(kind=dp), dimension(:,:), intent(out) :: grad_out
        real(kind=dp), dimension(:,:), intent(in) :: dist
        real(kind=dp), intent(in) :: h

        integer i, j
        !write(*,*) omp_get_num_threads()
        !omp parallel private(i,j) num_threads(N_THREADS)
        !$omp parallel private(i,j)
        !$omp do 
        do i = 1, size(x,2)
            do j = 1, size(x,2)
                grad_out(:,i) = grad_out(:,i) + (grad(:,j)+(x(:,i)-x(:,j))/h**2)*exp(-dist(j,i)/(2*h**2)) 
            enddo
            grad_out(:,i) = grad_out(:,i)/size(x,2)
        enddo
        !$omp end do
        !$omp end parallel

    end subroutine

    subroutine rbf_ksd(x, grad, dist, h, ksd)
        implicit none
        real(kind=dp), dimension(:,:), intent(in) :: x
        real(kind=dp), dimension(:,:), intent(in) :: grad
        real(kind=dp), dimension(:,:), intent(in) :: dist
        real(kind=dp), intent(in) :: h
        real(kind=dp), intent(out) :: ksd

        real(kind=dp) :: kvalue
        real(kind=dp) :: trace
        integer i, j, n

        ksd = 0
        n = size(x,2)
        !$omp parallel private(i,j,kvalue) reduction(+:ksd)
        !$omp do 
        do i = 1, n
            do j = i+1, n
                kvalue = exp(-dist(j,i)/(2*h**2))
                ksd = ksd + kvalue*sum(grad(:,i)*grad(:,j)) + &
                    sum(grad(:,i)*(x(:,i)-x(:,j))/h**2*kvalue) + &
                    sum(-(x(:,i)-x(:,j))/h**2*kvalue*grad(:,j)) + &
                    (size(x,1)-sum((x(:,i)-x(:,j))**2)/h**2)/h**2*kvalue
            enddo
        enddo
        !$omp end do
        !$omp end parallel
        ksd = ksd*2/(n*(n-1))

    end subroutine

    subroutine diagonal_kernel(x, grad, w, dist, h, grad_out)
        implicit none
        real(kind=dp), dimension(:,:), intent(in) :: x
        real(kind=dp), dimension(:,:), intent(in) :: grad
        real(kind=dp), dimension(:,:), intent(out) :: grad_out
        real(kind=dp), dimension(:), intent(in) :: w
        real(kind=dp), dimension(:,:), intent(in) :: dist
        real(kind=dp), intent(in) :: h

        integer i, j
        !omp parallel private(i,j) num_threads(N_THREADS)
        !$omp parallel private(i,j)
        !$omp do 
        do i = 1, size(x,2)
            do j = 1, size(x,2)
                grad_out(:,i) = grad_out(:,i) + (grad(:,j)/w+(x(:,i)-x(:,j))/h**2)*exp(-dist(j,i)/(2*h**2)) 
            enddo
            grad_out(:,i) = grad_out(:,i)/size(x,2)
        enddo
        !$omp end do
        !$omp end parallel

    end subroutine

    subroutine diagonal_ksd(x, grad, w, dist, h, ksd)
        implicit none
        real(kind=dp), dimension(:,:), intent(in) :: x
        real(kind=dp), dimension(:,:), intent(in) :: grad
        real(kind=dp), dimension(:), intent(in) :: w
        real(kind=dp), dimension(:,:), intent(in) :: dist
        real(kind=dp), intent(in) :: h
        real(kind=dp), intent(out) :: ksd

        real(kind=dp) :: kvalue
        real(kind=dp) :: trace
        integer i, j, n

        ksd = 0
        n = size(x,2)
        !$omp parallel private(i,j,kvalue) reduction(+:ksd)
        !$omp do 
        do i = 1, n
            do j = i+1, n
                kvalue = exp(-dist(j,i)/(2*h**2))
                ksd = ksd + kvalue*sum(grad(:,i)*grad(:,j)/w) + &
                    sum(grad(:,i)*(x(:,i)-x(:,j))/h**2*kvalue) + &
                    sum(-(x(:,i)-x(:,j))/h**2*kvalue*grad(:,j)) + &
                    (size(x,1)-dist(j,i)/h**2)/h**2*kvalue
            enddo
        enddo
        !$omp end do
        !$omp end parallel
        ksd = ksd*2/(n*(n-1))

    end subroutine

    subroutine svgd_grad(m, n, x, grad, kernel, dist, hessian, h, grad_out) bind(c)
        implicit none
        integer(c_int), intent(in), value :: m, n
        real(kind=dp), intent(in) :: x(n,m)
        real(kind=dp), intent(in) :: grad(n,m)
        integer(c_int), intent(in), value :: kernel
        real(kind=dp), intent(in) :: dist(m,m)
        real(kind=dp), intent(in) :: hessian(n)
        real(kind=dp), intent(in), value :: h
        real(kind=dp), intent(out) :: grad_out(n,m)
        
        select case(kernel)
        case (1)
            call rbf_kernel(x, grad, dist, h, grad_out)
        case (2)
            call diagonal_kernel(x, grad, hessian, dist, h, grad_out)
        case default
            write(*,*) "Error: not supported kernel."
        end select

    end subroutine

    subroutine ksd(m, n, x, grad, kernel, dist, hessian, h, ksd_value) bind(c)
        implicit none
        integer(c_int), intent(in), value :: m, n
        real(kind=dp), intent(in) :: x(n,m)
        real(kind=dp), intent(in) :: grad(n,m)
        integer(c_int), intent(in), value :: kernel
        real(kind=dp), intent(in) :: dist(m,m)
        real(kind=dp), intent(in) :: hessian(n)
        real(kind=dp), intent(in), value :: h
        real(kind=dp), intent(out) :: ksd_value
        
        real(kind=dp) :: kvalue
        select case(kernel)
        case (1)
            call rbf_ksd(x, grad, dist, h, kvalue)
        case (2)
            call diagonal_ksd(x, grad, hessian, dist, h, kvalue)
        case default
            write(*,*) "Error: not supported kernel."
        end select
        ksd_value = kvalue

    end subroutine

end module
