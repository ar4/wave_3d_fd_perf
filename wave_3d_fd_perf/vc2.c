#define A(a, x, y, z)  (a[(z) * ny * nx + (y) * nx + x])

void step(float *restrict f,
                float *restrict fp,
                const int nx,
                const int ny,
                const int nz,
                const int nxi,
                const float *restrict const model_padded2_dt2,
                const float dx,
                const float dt,
                const float *restrict const sources,
                const int *restrict const sources_x,
                const int *restrict const sources_y,
                const int *restrict const sources_z,
                const int num_sources, const int source_len, const int num_steps)
{

        int step;
        int x;
        int y;
        int z;
        int i;
        int sx;
        int sy;
        int sz;
        float f_xx;
        float *tmp;
        float fd_coeff[9] = {
                -924708642.0f / 302702400 / (dx * dx),
                538137600.0f / 302702400 / (dx * dx),
                -94174080.0f / 302702400 / (dx * dx),
                22830080.0f / 302702400 / (dx * dx),
                -5350800.0f / 302702400 / (dx * dx),
                1053696.0f / 302702400 / (dx * dx),
                -156800.0f / 302702400 / (dx * dx),
                15360.0f / 302702400 / (dx * dx),
                -735.0f / 302702400 / (dx * dx)
        };

        for (step = 0; step < num_steps; step++) {
                for (z = 8; z < nz - 8; z++) {
                        for (y = 8; y < ny - 8; y++) {
                                for (x = 8; x < nxi + 8; x++) {
                                        f_xx = 3 * fd_coeff[0] * A(f, x, y, z) +
                                                fd_coeff[1] *
                                                (A(f, x + 1, y, z) + 
                                                 A(f, x - 1, y, z) +
                                                 A(f, x, y + 1, z) +
                                                 A(f, x, y - 1, z) +
                                                 A(f, x, y, z + 1) +
                                                 A(f, x, y, z - 1)) +
                                                fd_coeff[2] *
                                                (A(f, x + 2, y, z) + 
                                                 A(f, x - 2, y, z) +
                                                 A(f, x, y + 2, z) +
                                                 A(f, x, y - 2, z) +
                                                 A(f, x, y, z + 2) +
                                                 A(f, x, y, z - 2)) +
                                                fd_coeff[3] *
                                                (A(f, x + 3, y, z) + 
                                                 A(f, x - 3, y, z) +
                                                 A(f, x, y + 3, z) +
                                                 A(f, x, y - 3, z) +
                                                 A(f, x, y, z + 3) +
                                                 A(f, x, y, z - 3)) +
                                                fd_coeff[4] *
                                                (A(f, x + 4, y, z) + 
                                                 A(f, x - 4, y, z) +
                                                 A(f, x, y + 4, z) +
                                                 A(f, x, y - 4, z) +
                                                 A(f, x, y, z + 4) +
                                                 A(f, x, y, z - 4)) +
                                                fd_coeff[5] *
                                                (A(f, x + 5, y, z) + 
                                                 A(f, x - 5, y, z) +
                                                 A(f, x, y + 5, z) +
                                                 A(f, x, y - 5, z) +
                                                 A(f, x, y, z + 5) +
                                                 A(f, x, y, z - 5)) +
                                                fd_coeff[6] *
                                                (A(f, x + 6, y, z) + 
                                                 A(f, x - 6, y, z) +
                                                 A(f, x, y + 6, z) +
                                                 A(f, x, y - 6, z) +
                                                 A(f, x, y, z + 6) +
                                                 A(f, x, y, z - 6)) +
                                                fd_coeff[7] *
                                                (A(f, x + 7, y, z) + 
                                                 A(f, x - 7, y, z) +
                                                 A(f, x, y + 7, z) +
                                                 A(f, x, y - 7, z) +
                                                 A(f, x, y, z + 7) +
                                                 A(f, x, y, z - 7)) +
                                                fd_coeff[8] *
                                                (A(f, x + 8, y, z) + 
                                                 A(f, x - 8, y, z) +
                                                 A(f, x, y + 8, z) +
                                                 A(f, x, y - 8, z) +
                                                 A(f, x, y, z + 8) +
                                                 A(f, x, y, z - 8));

                                        A(fp, x, y, z) =
                                                A(model_padded2_dt2, x, y, z) *
                                                f_xx +
                                                2 * A(f, x, y, z) -
                                                A(fp, x, y, z);
                                }
                        }
                }

                for (i = 0; i < num_sources; i++) {
                        sx = sources_x[i] + 8;
                        sy = sources_y[i] + 8;
                        sz = sources_z[i] + 8;
                        A(fp, sx, sy, sz) +=
                                A(model_padded2_dt2, sx, sy, sz) *
                                sources[i * source_len + step] * dt;
                }

                tmp = f;
                f = fp;
                fp = tmp;
        }
}
