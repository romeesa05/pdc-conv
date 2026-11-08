# Both programs are compiled with exactly the same flags.
CC     = gcc
CFLAGS = -O3 -fopenmp
LIBS   = -lm

all: serial parallel

serial: src/serial.c
	$(CC) $(CFLAGS) src/serial.c -o serial $(LIBS)

parallel: src/parallel.c
	$(CC) $(CFLAGS) src/parallel.c -o parallel $(LIBS)

clean:
	rm -f serial parallel

# delete all saved images and results
reset:
	rm -rf output results
