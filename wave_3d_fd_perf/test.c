#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <stdbool.h>

void step(
                float *restrict f,
                float *restrict fp,
                const int nx,
                const int ny,
                const int nz,
                const int nxi,
                const float *restrict const model_padded2_dt2,
                const float dx, const float dt,
                const float *restrict const sources,
                const int *restrict const sources_x,
                const int *restrict const sources_y,
                const int *restrict const sources_z,
                const int num_sources, const int source_len,
                const int num_steps);

int main(const int argc, const char *restrict const argv[])
{

        float *f;
        float *fp;
        float *model2_dt2;
        float *sources;
        int *sources_x;
        int *sources_y;
        int *sources_z;
        const int nx = atoi(argv[1]);
        const int ny = atoi(argv[2]);
        const int nz = atoi(argv[3]);
        const bool align = argc > 4;
        const int align_val = align ? atoi(argv[4]) : 1;
        const int ny_pad = ny + 16;
        const int nz_pad = nz + 16;
        const int nx_pad =
                align ? (int)ceilf((float)(nx + 16) / align_val) * align_val :
                nx + 16;
        const int num_steps = 10;
        const int source_len = num_steps;
        const int num_sources = 1;
        int source_idx;

        if (align) {
                f = aligned_alloc(align_val, sizeof(float) *
                                nx_pad * ny_pad * nz_pad);
                fp = aligned_alloc(align_val, sizeof(float) *
                                nx_pad * ny_pad * nz_pad);
                model2_dt2 =
                        aligned_alloc(align_val, sizeof(float) *
                                        nx_pad * ny_pad * nz_pad);
                sources =
                        aligned_alloc(align_val,
                                        sizeof(float) * num_sources *
                                        source_len);
                sources_x = aligned_alloc(align_val, sizeof(int) * num_sources);
                sources_y = aligned_alloc(align_val, sizeof(int) * num_sources);
                sources_z = aligned_alloc(align_val, sizeof(int) * num_sources);
        } else {
                f = malloc(sizeof(float) * nx_pad * ny_pad * nz_pad);
                fp = malloc(sizeof(float) * nx_pad * ny_pad * nz_pad);
                model2_dt2 = malloc(sizeof(float) * nx_pad * ny_pad * nz_pad);
                sources = malloc(sizeof(float) * num_sources * source_len);
                sources_x = malloc(sizeof(int) * num_sources);
                sources_y = malloc(sizeof(int) * num_sources);
                sources_z = malloc(sizeof(int) * num_sources);
        }

        if (f == NULL || fp == NULL || model2_dt2 == NULL || sources == NULL
                        || sources_x == NULL || sources_y == NULL) {
                free(f);
                free(fp);
                free(model2_dt2);
                free(sources);
                free(sources_x);
                free(sources_y);
                free(sources_z);
                return EXIT_FAILURE;
        }

        memset(f, 0, sizeof(float) * nx_pad * ny_pad * nz_pad);
        memset(fp, 0, sizeof(float) * nx_pad * ny_pad * nz_pad);
        memset(model2_dt2, 0, sizeof(float) * nx_pad * ny_pad * nz_pad);
        memset(sources, 0, sizeof(float) * num_sources * source_len);
        memset(sources_x, 0, sizeof(int) * num_sources);
        memset(sources_y, 0, sizeof(int) * num_sources);
        memset(sources_z, 0, sizeof(int) * num_sources);

        for (source_idx = 0; source_idx < num_sources; source_idx++) {
                sources_x[source_idx] = source_idx;
                sources_y[source_idx] = source_idx;
                sources_z[source_idx] = source_idx;
        }

        step(f, fp, nx_pad, ny_pad, nz_pad, nx, model2_dt2, 5.0f, 0.0001f,
                        sources, sources_x, sources_y, sources_z,
                        num_sources, source_len, num_steps);

        free(f);
        free(fp);
        free(model2_dt2);
        free(sources);
        free(sources_x);
        free(sources_y);
        free(sources_z);

        return EXIT_SUCCESS;

}
