/*
 * parallel.c  -  2D image convolution, PARALLEL version (OpenMP)
 *
 * Usage:
 *   ./parallel [size|image_file] [filter] [threads] [schedule]
 *
 *   size        : makes a synthetic size x size grayscale image (default 2048)
 *   image_file  : or load a real image (png/jpg), converted to grayscale
 *   filter      : blur | sharpen | edge | sobel | gaussian5 | edge5 | all
 *                 (default all)
 *   threads     : number of threads (default: all logical processors)
 *   schedule    : static | dynamic | guided  (default static)
 *
 * Examples:
 *   ./parallel 4096 gaussian5 8
 *   ./parallel 4096 all 4 dynamic
 *
 * Prints image size, mode, threads and execution time. If ./serial was run
 * before with the same image and filter, it also prints the speedup and
 * efficiency and checks that the output is identical to the serial output.
 * Saves the result in output/parallel/ and adds a line to results/results.csv
 *
 * The ONLY difference from serial.c is the "#pragma omp parallel for" line in
 * convolve(). Everything else is the same code.
 */
#include <math.h>
#include <omp.h>
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
/* Filters (same as serial.c)                                              */
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
/* Synthetic test image (same as serial.c, so both get identical input)    */
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
/* The convolution - the rows of the image are divided among the threads.  */
/*                                                                         */
/* Why this is safe (no race condition): every thread only READS the input */
/* image and WRITES its own rows of the output image. No two threads ever  */
/* write the same pixel, so no locks or atomics are needed.                */
/* sum1, sum2, x, i, j, ... are declared inside the loop, so each thread   */
/* has its own private copy.                                               */
/* ----------------------------------------------------------------------- */
void convolve(const unsigned char *in, unsigned char *out, int W, int H,
              const Filter *f)
{
    int r = f->size / 2;

    #pragma omp parallel for schedule(runtime)
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

/* Find the most recent serial time for this image/filter in results.csv.
 * Returns 0 if ./serial was not run for it yet. */
double find_serial_time(const char *source, const char *filter, int W, int H)
{
    FILE *fp = fopen("results/results.csv", "r");
    if (!fp) return 0;
    char line[512], mode[32], img[128], fil[32];
    int w, h;
    double t, found = 0;
    while (fgets(line, sizeof line, fp)) {
        if (sscanf(line, "%31[^,],%127[^,],%31[^,],%d,%d,%*d,%*[^,],%lf",
                   mode, img, fil, &w, &h, &t) == 6 &&
            !strcmp(mode, "serial") && !strcmp(img, source) &&
            !strcmp(fil, filter) && w == W && h == H)
            found = t;
    }
    fclose(fp);
    return found;
}

/* Compare with the image saved by ./serial. Returns "yes", "NO" or "-". */
const char *check_correct(const char *serial_path, const unsigned char *out,
                          int W, int H, int *maxdiff)
{
    int w, h, n;
    unsigned char *ref = stbi_load(serial_path, &w, &h, &n, 1);
    *maxdiff = -1;
    if (!ref) return "-";
    if (w != W || h != H) { stbi_image_free(ref); return "-"; }
    int m = 0;
    for (size_t i = 0; i < (size_t)W * H; i++) {
        int d = abs(ref[i] - out[i]);
        if (d > m) m = d;
    }
    stbi_image_free(ref);
    *maxdiff = m;
    return m == 0 ? "yes" : "NO";
}

/* ----------------------------------------------------------------------- */
int main(int argc, char **argv)
{
    const char *arg1 = argc > 1 ? argv[1] : "2048";
    const char *which = argc > 2 ? argv[2] : "all";
    int threads = argc > 3 ? atoi(argv[3]) : omp_get_num_procs();
    const char *sched = argc > 4 ? argv[4] : "static";
    int W, H;
    unsigned char *in;
    char source[256];

    /* choose number of threads and loop schedule */
    if (threads < 1) threads = 1;
    omp_set_num_threads(threads);
    if      (!strcmp(sched, "dynamic")) omp_set_schedule(omp_sched_dynamic, 0);
    else if (!strcmp(sched, "guided"))  omp_set_schedule(omp_sched_guided, 0);
    else { sched = "static";            omp_set_schedule(omp_sched_static, 0); }

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
    mkdir("output/parallel", 0755);
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
        char path[512], serial_path[512];
        snprintf(path, sizeof path, "output/parallel/%s_%dx%d_%s_%dthreads.png",
                 source, W, H, f->name, threads);
        stbi_write_png_compression_level = 1;
        stbi_write_png(path, W, H, 1, out, W);

        /* 4. compare with the serial version */
        snprintf(serial_path, sizeof serial_path, "output/serial/%s_%dx%d_%s.png",
                 source, W, H, f->name);
        double t_serial = find_serial_time(source, f->name, W, H);
        int maxdiff;
        const char *correct = check_correct(serial_path, out, W, H, &maxdiff);

        /* 5. print and store the result */
        printf("----------------------------------------\n");
        printf("Image size      : %d x %d (%s)\n", W, H, source);
        printf("Processing mode : Parallel (OpenMP, %d threads, %s schedule)\n", threads, sched);
        printf("Filter          : %s (%dx%d)\n", f->name, f->size, f->size);
        printf("Execution time  : %.4f s (average of %d runs)\n", t, RUNS);
        if (t_serial > 0) {
            printf("Serial time     : %.4f s\n", t_serial);
            printf("Speedup         : %.2fx\n", t_serial / t);
            printf("Efficiency      : %.0f%%\n", 100 * t_serial / t / threads);
        } else {
            printf("Speedup         : run ./serial %s %s first\n", arg1, f->name);
        }
        if (maxdiff >= 0)
            printf("Same as serial  : %s (max pixel difference %d)\n", correct, maxdiff);
        printf("Saved image     : %s\n", path);

        if (t_serial > 0)
            fprintf(csv, "parallel,%s,%s,%d,%d,%d,%s,%.6f,%.2f,%.2f,%s\n", source, f->name,
                    W, H, threads, sched, t, t_serial / t, t_serial / t / threads, correct);
        else
            fprintf(csv, "parallel,%s,%s,%d,%d,%d,%s,%.6f,-,-,%s\n", source, f->name,
                    W, H, threads, sched, t, correct);
    }
    if (!found) printf("Unknown filter '%s'\n", which);

    fclose(csv);
    free(in);
    free(out);
    return 0;
}
