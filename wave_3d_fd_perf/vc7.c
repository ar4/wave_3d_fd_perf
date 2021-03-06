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
		-924708642.0f / 302702400 / (dx * dx) / 2,
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
					f_xx = 0.0f;
					for (i = 0; i < 9; i++) {
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
					    f_xx +
					    2 * A(f, x, y, z) - A(fp, x, y, z);
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
