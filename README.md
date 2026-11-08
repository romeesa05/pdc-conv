# Parallel 2D Image Convolution with OpenMP

Two programs:

| File | What it does |
|---|---|
| `src/serial.c` | convolution with 1 thread (plain loops) |
| `src/parallel.c` | the same code, plus one line, `#pragma omp parallel for`, which splits the image rows among the threads |

Each run prints the image size, processing mode, filter and execution time.
It saves the result image in `output/serial/` or `output/parallel/` and adds a
line to **`results/results.csv`**.

The parallel program also prints the **speedup** and **efficiency**, which it
calculates from the serial time already in `results.csv`. It also checks that
its image is **identical** to the serial image.

Filters: `blur`, `sharpen`, `edge`, `sobel` (3×3) and `gaussian5`, `edge5` (5×5), or `all`.

## How to run (in WSL Ubuntu)

```bash
cd "/mnt/c/Users/amir/Downloads/pdc-conv/pdc-conv"
make                              # builds ./serial and ./parallel
```

```bash
./serial 4096 gaussian5           # size 4096x4096, one filter
./parallel 4096 gaussian5 8       # same, 8 threads
./parallel 4096 gaussian5 8 dynamic   # schedule: static | dynamic | guided
./serial 4096 all                 # every filter
./serial data/photo.jpg sobel     # a real image instead of a synthetic one
```

Always run `./serial` first for a size and filter. `./parallel` needs that
serial result to calculate the speedup and check correctness.

Run all the experiments and make the graphs (about 20 minutes):

```bash
bash run_all.sh
```

This runs sizes 1024 to 16384 with 1, 2, 4, 8 and 16 threads, plus the
static/dynamic/guided comparison. The graphs go to `results/graphs/`.

To start again from nothing, run `make reset`. It deletes `output/` and `results/`.

## Example output

```
Image size      : 4096 x 4096 (synthetic)
Processing mode : Parallel (OpenMP, 8 threads, static schedule)
Filter          : sobel (3x3)
Execution time  : 0.0828 s (average of 3 runs)
Serial time     : 0.3840 s
Speedup         : 4.64x
Efficiency      : 58%
Same as serial  : yes (max pixel difference 0)
Saved image     : output/parallel/synthetic_4096x4096_sobel_8threads.png
```

## Notes for the report

* **Timing** covers only the convolution (`omp_get_wtime`), averaged over 3
  runs. Image creation and saving are not timed.
* **No race conditions.** Threads only read the input image, and each thread
  writes its own rows of the output, so no two threads write the same pixel.
* **Borders.** A pixel outside the image takes the value of the nearest edge pixel.
* **Speedup stops growing after about 4–8 threads.** The laptop (i5-1155G7)
  has 4 real cores with 2 hardware threads each. 16 threads is more threads
  than the CPU has, and convolution is also limited by memory speed.
