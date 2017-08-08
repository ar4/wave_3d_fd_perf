module vf1

        implicit none

contains

        subroutine step(f1, f2, model_padded2_dt2, nxi, dx, dt,        &
                        sources, sources_x, sources_y, sources_z,      &
                        num_steps)

                real, intent (in out), dimension (:, :, :) :: f1
                real, intent (in out), dimension (:, :, :) :: f2
                real, intent (in), dimension (:, :, :) ::              &
                        model_padded2_dt2
                integer, intent (in) :: nxi
                real, intent (in) :: dx
                real, intent (in) :: dt
                real, intent (in), dimension (:, :) :: sources
                integer, intent (in), dimension (:) :: sources_x
                integer, intent (in), dimension (:) :: sources_y
                integer, intent (in), dimension (:) :: sources_z
                integer, intent (in) :: num_steps

                integer :: step_idx
                logical :: even
                real, dimension(9) :: fd_coeff

                fd_coeff = (/                                          &
                        -924708642.0 / 302702400 / (dx * dx),          &
                        538137600.0 / 302702400 / (dx * dx),           &
                        -94174080.0 / 302702400 / (dx * dx),           &
                        22830080.0 / 302702400 / (dx * dx),            &
                        -5350800.0 / 302702400 / (dx * dx),            &
                        1053696.0 / 302702400 / (dx * dx),             &
                        -156800.0 / 302702400 / (dx * dx),             &
                        15360.0 / 302702400 / (dx * dx),               &
                        -735.0 / 302702400 / (dx * dx)                 &
                        /)


                do step_idx = 1, num_steps
                even = (mod (step_idx, 2) == 0)
                if (even) then
                        call step_inner(f2, f1, model_padded2_dt2, nxi,&
                                dt, sources, sources_x, sources_y,     &
                                sources_z, step_idx, fd_coeff)
                else
                        call step_inner(f1, f2, model_padded2_dt2, nxi,&
                                dt, sources, sources_x, sources_y,     &
                                sources_z, step_idx, fd_coeff)
                end if
                end do

        end subroutine step


        subroutine step_inner(f, fp, model_padded2_dt2, nxi, dt,       &
                        sources, sources_x, sources_y, sources_z,      &
                        step_idx, fd_coeff)

                real, intent (in), dimension (:, :, :) :: f
                real, intent (in out), dimension (:, :, :) :: fp
                real, intent (in), dimension (:, :, :) ::              &
                        model_padded2_dt2
                integer, intent (in) :: nxi
                real, intent (in) :: dt
                real, intent (in), dimension (:, :) :: sources
                integer, intent (in), dimension (:) :: sources_x
                integer, intent (in), dimension (:) :: sources_y
                integer, intent (in), dimension (:) :: sources_z
                integer, intent (in) :: step_idx
                real, intent (in), dimension (9) :: fd_coeff

                integer :: x
                integer :: y
                integer :: z
                integer :: i
                integer :: sx
                integer :: sy
                integer :: sz
                integer :: ny
                integer :: nz
                integer :: num_sources
                real :: f_xx

                ny = size(f, dim=2)
                nz = size(f, dim=3)
                num_sources = size(sources, dim=2)

                do z = 9, nz - 8
                do y = 9, ny - 8
                do x = 9, nxi + 8
                f_xx = 3 * fd_coeff(1) * f(x, y, z)
                do i = 1, 8
                f_xx = f_xx + fd_coeff(i + 1) *                        &
                        (f(x + i, y, z) + f(x - i, y, z) +             &
                        f(x, y + i, z) + f(x, y - i, z) +              &
                        f(x, y, z + i) + f(x, y, z - i))
                end do
                fp(x, y, z) = (model_padded2_dt2(x, y, z) * f_xx +     &
                        2 * f(x, y, z) - fp(x, y, z))
                end do
                end do
                end do

                do i = 1, num_sources
                sx = sources_x(i) + 9
                sy = sources_y(i) + 9
                sz = sources_z(i) + 9
                fp(sx, sy, sz) = fp(sx, sy, sz) +                      &
                        model_padded2_dt2(sx, sy, sz) *                &
                        sources(step_idx, i) * dt
                end do

        end subroutine step_inner

end module vf1
