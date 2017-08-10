#include <immintrin.h>
#include <stdbool.h>
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
	const int last_avx_x = ((nxi - 1) / 8) * 8 + 8;

	float mask[8];
	for (i = 0; i < 8; i++) {
		if (last_avx_x + i < nxi + 8) {
			mask[i] = 1.0f;
		}
		else {
			mask[i] = 0.0f;
		}
	}
	const __m256 ymask = _mm256_set_ps(mask[7], mask[6], mask[5], mask[4], mask[3], mask[2], mask[1], mask[0]);

	const __m256 ytwo = _mm256_set1_ps(2.0f);
	for (step = 0; step < num_steps; step++) {
#pragma omp parallel for default(none) private(y, x, i) shared(fd_coeff, f, fp)
		for (z = 8; z < nz - 8; z++) {
			for (y = 8; y < ny - 8; y++) {
				for (x = 8; x < nxi; x+=8) {
					__m256 yf_xx = _mm256_setzero_ps();
					__m256 ymodel_padded2_dt2 = _mm256_loadu_ps(&A(model_padded2_dt2, x, y, z));
					for (i = 0; i < 9; i++) {
						__m256 yfd_coeff = _mm256_set1_ps(fd_coeff[i]);
						__m256 yfx1 = _mm256_loadu_ps(&A(f, x + i, y, z));
						__m256 yfx2 = _mm256_loadu_ps(&A(f, x - i, y, z));
						__m256 yfy1 = _mm256_loadu_ps(&A(f, x, y + i, z));
						__m256 yfy2 = _mm256_loadu_ps(&A(f, x, y - i, z));
						__m256 yfz1 = _mm256_loadu_ps(&A(f, x, y, z + i));
						__m256 yfz2 = _mm256_loadu_ps(&A(f, x, y, z - i));
						yf_xx = _mm256_fmadd_ps(yfx1, yfd_coeff, yf_xx);
						yf_xx = _mm256_fmadd_ps(yfx2, yfd_coeff, yf_xx);
						yf_xx = _mm256_fmadd_ps(yfy1, yfd_coeff, yf_xx);
						yf_xx = _mm256_fmadd_ps(yfy2, yfd_coeff, yf_xx);
						yf_xx = _mm256_fmadd_ps(yfz1, yfd_coeff, yf_xx);
						yf_xx = _mm256_fmadd_ps(yfz2, yfd_coeff, yf_xx);
					}
					yf_xx = _mm256_mul_ps(ymodel_padded2_dt2, yf_xx);
					__m256 yf = _mm256_loadu_ps(&A(f, x, y, z));
					__m256 yfp = _mm256_loadu_ps(&A(fp, x, y, z));
					yfp = _mm256_sub_ps(yf_xx, yfp);
					yfp = _mm256_fmadd_ps(yf, ytwo, yfp);

					_mm256_storeu_ps(&A(fp, x, y, z), yfp);
				}
				if (last_avx_x < nxi + 8) {
					x = last_avx_x;
					__m256 yf_xx = _mm256_setzero_ps();
					__m256 ymodel_padded2_dt2 = _mm256_loadu_ps(&A(model_padded2_dt2, x, y, z));
					for (i = 0; i < 9; i++) {
						__m256 yfd_coeff = _mm256_set1_ps(fd_coeff[i]);
						__m256 yfx1 = _mm256_loadu_ps(&A(f, x + i, y, z));
						__m256 yfx2 = _mm256_loadu_ps(&A(f, x - i, y, z));
						__m256 yfy1 = _mm256_loadu_ps(&A(f, x, y + i, z));
						__m256 yfy2 = _mm256_loadu_ps(&A(f, x, y - i, z));
						__m256 yfz1 = _mm256_loadu_ps(&A(f, x, y, z + i));
						__m256 yfz2 = _mm256_loadu_ps(&A(f, x, y, z - i));
						yf_xx = _mm256_fmadd_ps(yfx1, yfd_coeff, yf_xx);
						yf_xx = _mm256_fmadd_ps(yfx2, yfd_coeff, yf_xx);
						yf_xx = _mm256_fmadd_ps(yfy1, yfd_coeff, yf_xx);
						yf_xx = _mm256_fmadd_ps(yfy2, yfd_coeff, yf_xx);
						yf_xx = _mm256_fmadd_ps(yfz1, yfd_coeff, yf_xx);
						yf_xx = _mm256_fmadd_ps(yfz2, yfd_coeff, yf_xx);
					}
					yf_xx = _mm256_mul_ps(ymodel_padded2_dt2, yf_xx);
					__m256 yf = _mm256_loadu_ps(&A(f, x, y, z));
					__m256 yfp = _mm256_loadu_ps(&A(fp, x, y, z));
					yfp = _mm256_sub_ps(yf_xx, yfp);
					yfp = _mm256_fmadd_ps(yf, ytwo, yfp);

					yfp = _mm256_mul_ps(yfp, ymask);
					_mm256_storeu_ps(&A(fp, x, y, z), yfp);
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
