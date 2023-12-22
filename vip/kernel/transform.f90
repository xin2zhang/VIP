! Invertible transform
!
module m_transform

    use iso_c_binding
    use m_utils, only : dp, sp, N_THREADS
    implicit none
    real(kind=dp), parameter :: eps0 = 1e-30
    real(kind=dp), parameter :: eps = tiny(eps0)

contains

    ! transform constrained variable into infinite space
    subroutine transform(qvals, lower_bounds, upper_bounds)
        use ieee_arithmetic
        implicit none
        real(kind=dp), dimension(:), intent(inout) :: qvals
        real(kind=dp), dimension(:), intent(in) :: lower_bounds
        real(kind=dp), dimension(:), intent(in) :: upper_bounds

        integer i

        do i = 1, size(qvals)
            if(ieee_is_finite(lower_bounds(i)) .and. .not.ieee_is_finite(upper_bounds(i)))then
                qvals(i) = log(qvals(i)-lower_bounds(i))
            elseif(.not.ieee_is_finite(lower_bounds(i)) .and. ieee_is_finite(upper_bounds(i)))then
                qvals(i) = log(upper_bounds(i)-qvals(i))
            else
                qvals(i) = log(qvals(i)-lower_bounds(i)) - log(upper_bounds(i)-qvals(i))
            endif
        enddo
           
    endsubroutine transform

    subroutine inv_transform(qvals, lower_bounds, upper_bounds)
        use ieee_arithmetic
        implicit none
        real(kind=dp), dimension(:), intent(inout) :: qvals
        real(kind=dp), dimension(:), intent(in) :: lower_bounds
        real(kind=dp), dimension(:), intent(in) :: upper_bounds

        integer i
        real(kind=dp) exp_term

        do i = 1, size(qvals)
            if(ieee_is_finite(lower_bounds(i)) .and. .not.ieee_is_finite(upper_bounds(i)))then
                qvals(i) = lower_bounds(i) + exp(qvals(i))
            elseif(.not.ieee_is_finite(lower_bounds(i)) .and. ieee_is_finite(upper_bounds(i)))then
                qvals(i) = upper_bounds(i) - exp(qvals(i))
            else
                if(qvals(i)>=0)then
                    exp_term = exp(-qvals(i))
                    qvals(i) = lower_bounds(i) + (upper_bounds(i)-lower_bounds(i))&
                        / (1+exp_term)
                else
                    exp_term = exp(qvals(i))
                    qvals(i) = upper_bounds(i) + (lower_bounds(i) - upper_bounds(i))&
                        / (1+exp_term)
                endif
            endif
        enddo
           
    endsubroutine inv_transform

    function log_jacobian(qvals_trans, lower_bounds, upper_bounds)
        use ieee_arithmetic
        implicit none
        real(kind=dp), dimension(:), intent(in) :: qvals_trans
        real(kind=dp), dimension(:), intent(in) :: lower_bounds
        real(kind=dp), dimension(:), intent(in) :: upper_bounds
        real(kind=dp) log_jacobian

        integer i

        log_jacobian = 0.
        do i = 1, size(qvals_trans)
            if(ieee_is_finite(lower_bounds(i)) .and. .not.ieee_is_finite(upper_bounds(i)))then
                log_jacobian = log_jacobian + qvals_trans(i)
            elseif(.not.ieee_is_finite(lower_bounds(i)) .and. ieee_is_finite(upper_bounds(i)))then
                log_jacobian = log_jacobian + qvals_trans(i)
            else
                if(qvals_trans(i)>=0)then
                    log_jacobian = log_jacobian + log(upper_bounds(i)-lower_bounds(i)) &
                        - qvals_trans(i) - 2*log(1+exp(-qvals_trans(i)))
                else
                    log_jacobian = log_jacobian + log(upper_bounds(i)-lower_bounds(i)) &
                        + qvals_trans(i) - 2*log(1+exp(qvals_trans(i)))
                endif
            endif
        enddo

    endfunction log_jacobian

    ! calculate new gradient after transformation, note it is actually a diagonal matrix
    subroutine jacobian_trans_grad(grad, qvals_trans, lower_bounds, upper_bounds, mask)
        use ieee_arithmetic
        implicit none
        real(kind=dp), dimension(:), intent(inout) :: grad
        real(kind=dp), dimension(:), intent(in) :: qvals_trans
        real(kind=dp), dimension(:), intent(in) :: lower_bounds
        real(kind=dp), dimension(:), intent(in) :: upper_bounds
        integer(c_int), dimension(:), intent(out), optional :: mask

        integer i
        real(kind=dp) exp_term, gjacob, ggrad

        if(present(mask)) mask = 0

        do i = 1, size(qvals_trans)
            exp_term = exp(qvals_trans(i))
            if(ieee_is_finite(lower_bounds(i)) .and. .not.ieee_is_finite(upper_bounds(i)))then
                grad(i) = exp_term*grad(i) - 1
            elseif(.not.ieee_is_finite(lower_bounds(i)) .and. ieee_is_finite(upper_bounds(i)))then
                grad(i) = -exp_term*grad(i) - 1
            else
                if(qvals_trans(i)>=0)then
                    exp_term = exp(-qvals_trans(i))
                    gjacob = 1 - 2/(1+exp_term)
                else
                    exp_term = exp(qvals_trans(i))
                    gjacob = 1 - 2*exp_term/(1+exp_term)
                endif
                ggrad = exp_term*(upper_bounds(i)-lower_bounds(i))/(1+exp_term)**2 * grad(i)
                grad(i) = ggrad + gjacob
                if(present(mask) .and. abs(ggrad)<abs(gjacob))then
                    mask(i) = 1
                endif
            endif
        enddo

    endsubroutine jacobian_trans_grad

    subroutine many_transform(qvals, lower_bounds, upper_bounds)
        implicit none
        real(kind=dp), dimension(:,:), intent(inout) :: qvals
        real(kind=dp), dimension(:), intent(in) :: lower_bounds
        real(kind=dp), dimension(:), intent(in) :: upper_bounds

        integer n, i
        n = size(qvals,1)

        !omp parallel private(i) num_threads(N_THREADS)
        !$omp parallel private(i)
        !$omp do
        do i = 1, size(qvals,2)
            call transform(qvals(:,i),lower_bounds,upper_bounds)
        enddo
        !$omp end do
        !$omp end parallel

    endsubroutine

    subroutine many_inv_transform(qvals_trans, lower_bounds, upper_bounds)
        implicit none
        real(kind=dp), dimension(:,:), intent(inout) :: qvals_trans
        real(kind=dp), dimension(:), intent(in) :: lower_bounds
        real(kind=dp), dimension(:), intent(in) :: upper_bounds

        integer n, i
        n = size(qvals_trans,1)

        !omp parallel private(i) num_threads(N_THREADS)
        !$omp parallel private(i)
        !$omp do
        do i = 1, size(qvals_trans,2)
            call inv_transform(qvals_trans(:,i),lower_bounds,upper_bounds)
        enddo
        !$omp end do
        !$omp end parallel

    endsubroutine

    subroutine many_log_jacobian(qvals_trans, lower_bounds, upper_bounds, jac)
        implicit none
        real(kind=dp), dimension(:,:), intent(in) :: qvals_trans
        real(kind=dp), dimension(:), intent(in) :: lower_bounds
        real(kind=dp), dimension(:), intent(in) :: upper_bounds
        real(kind=dp), dimension(:), intent(out) :: jac

        integer n, i

        do i = 1, size(qvals_trans,2)
            jac(i) = log_jacobian(qvals_trans(:,i),lower_bounds,upper_bounds)
        enddo

    endsubroutine

    subroutine many_jacobian_trans_grad(grad, qvals_trans, lower_bounds, upper_bounds, mask)
        implicit none
        real(kind=dp), dimension(:,:), intent(inout) :: grad
        real(kind=dp), dimension(:,:), intent(in) :: qvals_trans
        real(kind=dp), dimension(:), intent(in) :: lower_bounds
        real(kind=dp), dimension(:), intent(in) :: upper_bounds
        integer(c_int), dimension(:,:), intent(out), optional :: mask

        integer n, i

        !omp parallel private(i) num_threads(N_THREADS)
        !$omp parallel private(i)
        if(present(mask))then
            !$omp do
            do i = 1, size(qvals_trans,2)
                call jacobian_trans_grad(grad(:,i),qvals_trans(:,i),lower_bounds,upper_bounds,mask(:,i))
            enddo
            !$omp end do
        else
            !$omp do
            do i = 1, size(qvals_trans,2)
                call jacobian_trans_grad(grad(:,i),qvals_trans(:,i),lower_bounds,upper_bounds)
            enddo
            !$omp end do
        endif
        !$omp end parallel

    endsubroutine

    subroutine transform_c(n, qvals, lower_bnds, upper_bnds) bind(c)
        implicit none
        integer(c_int), intent(in) :: n
        real(kind=dp), intent(inout) :: qvals(n)
        real(kind=dp), intent(in) :: lower_bnds(n), upper_bnds(n)
        call transform(qvals,lower_bnds,upper_bnds)
    endsubroutine

    subroutine inv_transform_c(n, qvals, lower_bnds, upper_bnds) bind(c)
        implicit none
        integer(c_int), intent(in) :: n
        real(kind=dp), intent(inout) :: qvals(n)
        real(kind=dp), intent(in) :: lower_bnds(n), upper_bnds(n)
        call inv_transform(qvals,lower_bnds,upper_bnds)
    endsubroutine

    subroutine log_jacobian_c(n, qvals, lower_bnds, upper_bnds, jac) bind(c)
        implicit none
        integer(c_int), intent(in) :: n
        real(kind=dp), intent(in) :: qvals(n)
        real(kind=dp), intent(in) :: lower_bnds(n), upper_bnds(n)
        real(kind=dp), intent(out) :: jac
        jac=log_jacobian(qvals,lower_bnds,upper_bnds)
    endsubroutine

    subroutine trans_grad_c(n, grad, qvals, lower_bnds, upper_bnds) bind(c)
        implicit none
        integer(c_int), intent(in) :: n
        real(kind=dp), intent(inout) :: grad(n)
        real(kind=dp), intent(in) :: qvals(n)
        real(kind=dp), intent(in) :: lower_bnds(n), upper_bnds(n)
        call jacobian_trans_grad(grad,qvals,lower_bnds,upper_bnds)
    endsubroutine

    subroutine many_transform_c(m, n, qvals, lower_bnds, upper_bnds) bind(c)
        implicit none
        integer(c_int), intent(in) :: m, n
        real(kind=dp), intent(inout) :: qvals(n,m)
        real(kind=dp), intent(in) :: lower_bnds(n), upper_bnds(n)
        call many_transform(qvals,lower_bnds,upper_bnds)
    endsubroutine

    subroutine many_inv_transform_c(m, n, qvals, lower_bnds, upper_bnds) bind(c)
        implicit none
        integer(c_int), intent(in) :: m, n
        real(kind=dp), intent(inout) :: qvals(n,m)
        real(kind=dp), intent(in) :: lower_bnds(n), upper_bnds(n)
        call many_inv_transform(qvals,lower_bnds,upper_bnds)
    endsubroutine

    subroutine many_log_jacobian_c(m, n, qvals, lower_bnds, upper_bnds, jac) bind(c)
        implicit none
        integer(c_int), intent(in) :: m, n
        real(kind=dp), intent(in) :: qvals(n,m)
        real(kind=dp), intent(in) :: lower_bnds(n), upper_bnds(n)
        real(kind=dp), intent(out) :: jac(m)
        call many_log_jacobian(qvals,lower_bnds,upper_bnds,jac)
    endsubroutine

    subroutine many_trans_grad_c(m, n, grad, qvals, lower_bnds, upper_bnds) bind(c)
        implicit none
        integer(c_int), intent(in) :: m, n
        real(kind=dp), intent(inout) :: grad(n,m)
        real(kind=dp), intent(in) :: qvals(n,m)
        real(kind=dp), intent(in) :: lower_bnds(n), upper_bnds(n)
        call many_jacobian_trans_grad(grad,qvals,lower_bnds,upper_bnds)
    endsubroutine

    subroutine many_trans_grad_c2(m, n, grad, qvals, lower_bnds, upper_bnds, mask) bind(c)
        implicit none
        integer(c_int), intent(in) :: m, n
        real(kind=dp), intent(inout) :: grad(n,m)
        real(kind=dp), intent(in) :: qvals(n,m)
        real(kind=dp), intent(in) :: lower_bnds(n), upper_bnds(n)
        integer(kind=c_int), intent(inout) :: mask(n,m)
        call many_jacobian_trans_grad(grad,qvals,lower_bnds,upper_bnds,mask)
    endsubroutine

end module
