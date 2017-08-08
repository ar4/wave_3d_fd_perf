#define A(a, x, y, z)  (a[(z) * ny * nx + (y) * nx + x])

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
		  const float *restrict const fd_coeff, const int step)
{

	int x;
	int y;
	int z;
	int i;
	int sx;
	int sy;
	int sz;
	float f_xx;

#pragma omp parallel for default(none) private(y, x, f_xx, i)
	for (z = 8; z < nz - 8; z++) {
		for (y = 8; y < ny - 8; y++) {
			for (x = 8; x < nxi + 8; x++) {
				f_xx = 3 * fd_coeff[0] * A(f, x, y, z);
				for (i = 1; i < 9; i++) {
					f_xx += fd_coeff[i] *
					    (A(f, x + i, y, z) +
					     A(f, x - i, y, z) +
					     A(f, x, y + i, z) +
					     A(f, x, y - i, z) +
					     A(f, x, y, z + i) +
					     A(f, x, y, z - i));
				}

				A(fp, x, y, z) =
				    A(model_padded2_dt2, x, y, z) *
				    f_xx + 2 * A(f, x, y, z) - A(fp, x, y, z);
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

	for (step = 0; step < num_steps; step++) {
		inner(f, fp, nx, ny, nz, nxi, model_padded2_dt2, dt,
		      sources, sources_x, sources_y, sources_z,
		      num_sources, source_len, fd_coeff, step);

		tmp = f;
		f = fp;
		fp = tmp;
	}
}
