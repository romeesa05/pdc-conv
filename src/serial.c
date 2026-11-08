/*
 * serial.c  -  2D image convolution, SERIAL version (1 thread)
 *
 * Usage:
 *   ./serial [size|image_file] [filter]
 *
 *   size        : makes a synthetic size x size grayscale image (default 2048)
 *   image_file  : or load a real image (png/jpg), converted to grayscale
 *   filter      : blur | sharpen | edge | sobel | gaussian5 | edge5 | all
 *                 (default all)
 *
 * Examples:
 *   ./serial 4096 gaussian5
 *   ./serial data/photo.jpg sobel
 *
 * Prints image size, mode, filter and execution time, saves the result in
 * output/serial/ and adds a line to results/results.csv
 */
#include <math.h>
#include <omp.h>      /* only for omp_get_wtime(), no parallel code here */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image.h"
#include "stb_image_write.h"

#define RUNS 3        /* each filter is timed RUNS times, average reported */

/* ----------------------------------------------------------------------- */
/* Filters                                                                 */
/* ----------------------------------------------------------------------- */
enum { NORMAL, ABSOLUTE, MAGNITUDE };

typedef struct {
    const char *name;
    int size;        /* 3 or 5                                      */
    int div;         /* result is divided by this                   */
    int mode;        /* NORMAL, ABSOLUTE (edges), MAGNITUDE (sobel) */
    int k[25];       /* kernel                                      */
    int k2[25];      /* second kernel, only used by sobel (Gy)      */
} Filter;

Filter filters[] = {
    { "blur",      3, 9,   NORMAL,    { 1, 1, 1,
                                        1, 1, 1,
                                        1, 1, 1 }, {0} },
    { "sharpen",   3, 1,   NORMAL,    { 0,-1, 0,
                                       -1, 5,-1,
                                        0,-1, 0 }, {0} },
    { "edge",      3, 1,   ABSOLUTE,  { 0,-1, 0,
                                       -1, 4,-1,
                                        0,-1, 0 }, {0} },
    { "sobel",     3, 1,   MAGNITUDE, {-1, 0, 1,
                                       -2, 0, 2,
                                       -1, 0, 1 },
                                      {-1,-2,-1,
                                        0, 0, 0,
                                        1, 2, 1 } },
    { "gaussian5", 5, 256, NORMAL,    { 1, 4, 6, 4, 1,
                                        4,16,24,16, 4,
                                        6,24,36,24, 6,
                                        4,16,24,16, 4,
                                        1, 4, 6, 4, 1 }, {0} },
    { "edge5",     5, 1,   ABSOLUTE,  { 0, 0,-1, 0, 0,
                                        0,-1,-2,-1, 0,
                                       -1,-2,16,-2,-1,
                                        0,-1,-2,-1, 0,
                                        0, 0,-1, 0, 0 }, {0} },
};
int num_filters = sizeof(filters) / sizeof(filters[0]);

/* ----------------------------------------------------------------------- */
/* Synthetic test image: gradient + checkerboard + rings + noise.          */
/* Always the same pixels for the same size.                               */
/* ----------------------------------------------------------------------- */
void make_image(unsigned char *img, int W, int H)
{
    srand(42);
    int cell = W / 16 > 0 ? W / 16 : 1;
    for (int y = 0; y < H; y++) {
        for (int x = 0; x < W; x++) {
            int v = 160 * (x + y) / (W + H);                 /* gradient     */
            if (x < W / 2 && ((x / cell + y / cell) % 2))    /* checkerboard */
                v += 60;
            if (x >= W / 2) {                                /* rings        */
                double dx = x - 0.75 * W, dy = y - 0.5 * H;
                if ((int)(sqrt(dx * dx + dy * dy) / (W / 24.0 + 1)) % 2)
                    v += 70;
            }
            v += rand() % 25 - 12;                           /* noise        */
            img[(size_t)y * W + x] = v < 0 ? 0 : (v > 255 ? 255 : v);
        }
    }
}

int clamp(int v, int lo, int hi) { return v < lo ? lo : (v > hi ? hi : v); }

/* ----------------------------------------------------------------------- */
/* The convolution - plain nested loops, one thread.                       */
/* Pixels outside the image are replaced by the nearest edge pixel.        */
/* ----------------------------------------------------------------------- */
void convolve(const unsigned char *in, unsigned char *out, int W, int H,
              const Filter *f)
{
    int r = f->size / 2;

    for (int y = 0; y < H; y++) {
        for (int x = 0; x < W; x++) {
            int sum1 = 0, sum2 = 0;
            for (int i = -r; i <= r; i++) {
                int yy = clamp(y + i, 0, H - 1);
                for (int j = -r; j <= r; j++) {
                    int xx = clamp(x + j, 0, W - 1);
                    int p = in[(size_t)yy * W + xx];
                    int idx = (i + r) * f->size + (j + r);
                    sum1 += f->k[idx] * p;
                    sum2 += f->k2[idx] * p;
                }
            }
            int v;
            if (f->mode == MAGNITUDE) v = (int)(sqrt((double)sum1 * sum1 + (double)sum2 * sum2) + 0.5);
            else if (f->mode == ABSOLUTE) v = abs(sum1) / f->div;
            else v = (sum1 + f->div / 2) / f->div;
            out[(size_t)y * W + x] = clamp(v, 0, 255);
        }
    }
}

/* ----------------------------------------------------------------------- */
int main(int argc, char **argv)
{
    const char *arg1 = argc > 1 ? argv[1] : "2048";
    const char *which = argc > 2 ? argv[2] : "all";
    int W, H;
    unsigned char *in;
    char source[256];

    /* 1. get the input image (not timed) */
    if (atoi(arg1) > 0) {
        W = H = atoi(arg1);
        in = malloc((size_t)W * H);
        if (!in) { printf("Not enough memory for %d x %d\n", W, H); return 1; }
        make_image(in, W, H);
        strcpy(source, "synthetic");
    } else {
        int n;
        in = stbi_load(arg1, &W, &H, &n, 1);
        if (!in) { printf("Cannot open image %s\n", arg1); return 1; }
        const char *base = strrchr(arg1, '/') ? strrchr(arg1, '/') + 1 : arg1;
        snprintf(source, sizeof source, "%.*s", (int)strcspn(base, "."), base);
    }
    unsigned char *out = malloc((size_t)W * H);
    if (!out) { printf("Not enough memory\n"); return 1; }

    mkdir("output", 0755);          /* create the output folders */
    mkdir("output/serial", 0755);
    mkdir("results", 0755);
    FILE *csv = fopen("results/results.csv", "a");
    if (!csv) { printf("Cannot open results/results.csv\n"); return 1; }
    fseek(csv, 0, SEEK_END);
    if (ftell(csv) == 0)
        fprintf(csv, "mode,image,filter,width,height,threads,schedule,time_s,speedup,efficiency,correct\n");

    int found = 0;
    for (int fi = 0; fi < num_filters; fi++) {
        Filter *f = &filters[fi];
        if (strcmp(which, "all") != 0 && strcmp(which, f->name) != 0) continue;
        found = 1;

        /* 2. time only the convolution, RUNS times, take the average */
        double total = 0;
        for (int run = 0; run < RUNS; run++) {
            double t0 = omp_get_wtime();
            convolve(in, out, W, H, f);
            total += omp_get_wtime() - t0;
        }
        double t = total / RUNS;

        /* 3. save the result image */
        char path[512];
        snprintf(path, sizeof path, "output/serial/%s_%dx%d_%s.png", source, W, H, f->name);
        stbi_write_png_compression_level = 1;   /* fast saving for big images */
        stbi_write_png(path, W, H, 1, out, W);

        /* 4. print and store the result */
        printf("----------------------------------------\n");
        printf("Image size      : %d x %d (%s)\n", W, H, source);
        printf("Processing mode : Serial (1 thread)\n");
        printf("Filter          : %s (%dx%d)\n", f->name, f->size, f->size);
        printf("Execution time  : %.4f s (average of %d runs)\n", t, RUNS);
        printf("Saved image     : %s\n", path);

        fprintf(csv, "serial,%s,%s,%d,%d,1,-,%.6f,1.00,1.00,-\n", source, f->name, W, H, t);
    }
    if (!found) printf("Unknown filter '%s'\n", which);

    fclose(csv);
    free(in);
    free(out);
    return 0;
}
