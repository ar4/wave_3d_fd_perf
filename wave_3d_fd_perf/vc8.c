#define A(a, x, y, z)  (a[(z) * ny * nx + (y) * nx + x])

static void inner_block(const float *restrict const f,
			float *restrict const fp,
			const int nx,
			const int ny,
			const int nz,
			const int nxi,
			const float *restrict const model_padded2_dt2,
			const float dt,
			const float *restrict const fd_coeff,
			const int bx,
			const int by,
			const int bz,
			const int blocksize_x, const int blocksize_y,
			const int blocksize_z)
{

	int x;
	int y;
	int z;
	float f_xx;
	const int x_start = bx * blocksize_x + 8;
	const int y_start = by * blocksize_y + 8;
	const int z_start = bz * blocksize_z + 8;
	const int x_end = x_start + blocksize_x <= nxi + 8 ?
	    x_start + blocksize_x : nxi + 8;
	const int y_end = y_start + blocksize_y <= ny - 8 ?
	    y_start + blocksize_y : ny - 8;
	const int z_end = z_start + blocksize_z <= nz - 8 ?
	    z_start + blocksize_z : nz - 8;

	for (z = z_start; z < z_end; z++) {
		for (y = y_start; y < y_end; y++) {
			for (x = x_start; x < x_end; x++) {
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
				     A(f, x, y, z + 8) + A(f, x, y, z - 8));

				A(fp, x, y, z) =
				    A(model_padded2_dt2, x, y, z) *
				    f_xx + 2 * A(f, x, y, z) - A(fp, x, y, z);
			}
		}
	}
}

static void inner(const float *restrict const f,
		  float *restrict const fp,
		  const int nx,
		  const int ny,
		  const int nz,
		  const int nxi,
		  const float *restrict const model_padded2_dt2,
		  const float dt,
		  const float *restrict const sources,
		  const int *restrict const sources_x,
		  const int *restrict const sources_y,
		  const int *restrict const sources_z,
		  const int num_sources, const int source_len,
		  const float *restrict const fd_coeff, const int step,
		  const int blocksize_x, const int blocksize_y,
		  const int blocksize_z,
		  const int nbx, const int nby, const int nbz)
{

	int bx;
	int by;
	int bz;
	int i;
	int sx;
	int sy;
	int sz;

#pragma omp parallel for default(none) private(by, bx)
	for (bz = 0; bz < nbz; bz++) {
		for (by = 0; by < nby; by++) {
			for (bx = 0; bx < nbx; bx++) {
				inner_block(f, fp, nx, ny, nz, nxi,
					    model_padded2_dt2, dt, fd_coeff, bx,
					    by, bz, blocksize_x, blocksize_y,
					    blocksize_z);
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

}

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
	const int blocksize_x = 16;
	const int blocksize_y = 8;
	const int blocksize_z = 8;
	const int nbx = (int)((float)(nxi) / blocksize_x) +
	    (int)(((nxi) % blocksize_x) != 0);
	const int nby = (int)((float)(ny - 16) / blocksize_y) +
	    (int)(((ny - 16) % blocksize_y) != 0);
	const int nbz = (int)((float)(nz - 16) / blocksize_z) +
	    (int)(((nz - 16) % blocksize_z) != 0);

	for (step = 0; step < num_steps; step++) {
		inner(f, fp, nx, ny, nz, nxi, model_padded2_dt2, dt,
		      sources, sources_x, sources_y, sources_z,
		      num_sources, source_len, fd_coeff, step,
		      blocksize_x, blocksize_y, blocksize_z, nbx, nby, nbz);

		tmp = f;
		f = fp;
		fp = tmp;
	}
}
