# 1. Parallel Word Counting in a Text Corpus with MPI

Lab 2 - Estructura del Computador II. Static distribution by number of files (MPI v1) versus files sorted by size and dealt like cards (MPI v2).

## 2. Problem Description

`generator.py` (seed 42) creates the folder `dataset/` with:

- `consulta.txt`: 200 query words, one per line.
- 3000 corpus files `file_0001.txt` ... `file_3000.txt`, 44,951,458 words in total.

The file sizes are very different on purpose: about 60% small (2,000-5,000 words), 30% medium (10,000-20,000) and 10% large (50,000-100,000). The goal is to count how many times each query word appears in the whole corpus and to report the 10 most frequent ones.

Three programs solve the problem and must give exactly the same counts:

| File | What it is |
|---|---|
| `generator.py` | Dataset generator (provided, not modified) |
| `baseline_secuencial.py` | Sequential reference (provided, not modified) |
| `run_all.sh` | Script that runs everything (provided, not modified) |
| `detectar_procesos_mpi.py` | Script used to choose the number of processes p (provided in the previous lab) |
| `mpi1.py` | MPI version 1: broadcast + static distribution in contiguous blocks |
| `mpi2.py` | MPI version 2: files sorted by size and dealt like cards |

## 3. Environment and Execution Instructions

**Machine:** laptop with an Intel Core i5-12450H (8 cores: 4 performance cores with 2 threads each + 4 efficiency cores, 12 logical processors), 32 GB RAM, Windows 11, Docker Desktop 29.8.2. Everything runs inside the image `augustosalazar/slim-mpi:2` (Python 3.8.20, mpi4py 4.1.1, Open MPI 4.1.4), where `nproc` reports 12. The repository is in a normal local folder (not OneDrive) mounted into the container as `/app`.

All commands are run from the repository folder in PowerShell (`"$(pwd)"` on Linux/macOS, `"%cd%"` in cmd):

```powershell
# 1. Generate the dataset
docker run --rm -v "${PWD}:/app" augustosalazar/slim-mpi:2 python /app/generator.py

# 2. Sequential baseline (also writes dataset/baseline_results.csv)
docker run --rm -v "${PWD}:/app" augustosalazar/slim-mpi:2 python /app/baseline_secuencial.py

# 3. Choose p
docker run --rm -v "${PWD}:/app" augustosalazar/slim-mpi:2 python /app/detectar_procesos_mpi.py

# 4. MPI versions with p processes (p = 11, see section 4.a)
docker run --rm -v "${PWD}:/app" augustosalazar/slim-mpi:2 mpirun --allow-run-as-root --oversubscribe -np 11 python /app/mpi1.py
docker run --rm -v "${PWD}:/app" augustosalazar/slim-mpi:2 mpirun --allow-run-as-root --oversubscribe -np 11 python /app/mpi2.py

# 5. Everything at once with the provided script, only with p = 11
docker run --rm -e NP_LIST=11 -e RUNS=3 -v "${PWD}:/app" augustosalazar/slim-mpi:2 sh /app/run_all.sh
```

`.gitattributes` keeps `run_all.sh` with Linux line endings (LF), so it runs in the container without changes even when the repository is cloned on Windows.

### a. Experimental Plan

1. Generate the dataset once (fixed seed, so it is the same dataset as the instructor's).
2. Run the sequential baseline 6 times (4 before and 2 after the MPI runs) and record its time and its output. T_seq is the mean.
3. Run `detectar_procesos_mpi.py` and use the recommended N as p for every MPI run.
4. Run `mpi1.py` with p processes 5 times (the statement asks for at least 3). For each run record the total time and, for each process, the number of files and the local counting time t_i.
5. Compute T_p (mean total time), S_p = T_seq / T_p, E_p = S_p / p, LI = max(t_i) / mean(t_i) and the relative spread (max(t_i) - min(t_i)) / mean(t_i).
6. Write `mpi2.py` to reduce the imbalance seen in step 4, repeat steps 4-5 with it and compare both versions.
7. Check in every run that the result is identical to the baseline.

### b. Sequential baseline

`baseline_secuencial.py` reads `consulta.txt` into a set. Then it opens every `file_*.txt`, splits each line into words, converts them to lower case and adds 1 to a `Counter` when the word is in the set. It prints the time (measured with `time.perf_counter()` around the counting), the number of tokens and occurrences and the top 10, and it saves all the counts in `dataset/baseline_results.csv`.

### c. MPI version 1

`mpi1.py` follows the steps of the lab statement:

1. Rank 0 reads `consulta.txt`.
2. Rank 0 sends the query words to all processes with `comm.bcast`.
3. Rank 0 lists the `file_*.txt` files (sorted by name) and sends the list with `comm.bcast`.
4. Static distribution: the list is split into p contiguous blocks with the same number of files (272 or 273 for p = 11). Process r takes block r.
5. Each process counts the query words in its files with the same loop as the baseline, and measures its local time t_i with `MPI.Wtime()`.
6. `comm.gather` sends to rank 0 the partial counts, the number of files, the tokens read and t_i of every process.
7. Rank 0 adds the partial counts and prints the per-process report, `LOAD_IMBALANCE_LI`, `RELATIVE_SPREAD`, the top 10, `MATCHES_BASELINE` (whether the full table equals `baseline_results.csv`) and `EXECUTION_TIME`, the line that `run_all.sh` reads.

The total time starts after a `comm.Barrier()` (all processes start together) and stops when rank 0 has the final result, so it includes the broadcasts and the gather.

### d. Test procedure

- **Correctness:** every MPI run must print `MATCHES_BASELINE=True` and the same top 10 as the baseline.
- **Performance:** all runs are made in the same session, with no other heavy programs open. The MPI runs are interleaved (v1 run 1, v2 run 1, v1 run 2, v2 run 2, ...) so that any change in the state of the machine affects both versions in the same way. T_p is the mean of the 5 total times; the median is also reported because one slow run can move the mean a lot.
- The logs of every run are saved in `results/` (not uploaded, the folder is in `.gitignore`).

## 4. Experimental Plan Execution

### a. Sequential baseline timing

Runs 1-4 were made before the MPI runs and runs 5-6 right after them, to check that the machine did not change during the session.

| Run | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|
| Time (s) | 21.548 | 22.040 | 21.962 | 22.127 | 24.277 | 22.205 |

**T_seq = 22.360 s** (mean of the 6 runs, standard deviation 0.97 s; the median is 22.08 s). Run 5 is the slowest; without it the mean would be 21.97 s. Output: 3000 files, 44,951,458 tokens, 3,631,778 occurrences.

| Word | a | para | sus | otros | ante | unos | otra | vosotros | mios | tuya |
|---|---|---|---|---|---|---|---|---|---|---|
| Count | 785774 | 392156 | 228913 | 105530 | 99832 | 88794 | 83901 | 61617 | 58420 | 56635 |

It is the same top 10 as the example in the instructor's Readme.

**Choice of p.** `detectar_procesos_mpi.py` counts primes up to 2,000,000 with N = 1 to 12 processes (median of 3 runs for each N):

| N | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Median (s) | 3.685 | 2.233 | 2.342 | 1.618 | 1.639 | 1.670 | 1.421 | 1.211 | 1.462 | 1.311 | **1.180** | 1.921 |

The recommended value is N = 11, so **p = 11** in all MPI runs. N = 8, 10 and 11 are very close (1.18-1.31 s), so the choice of 11 instead of 8 is not strong.

### b. MPI version 1 timing results

p = 11. "Max t_i" is the local time of the slowest process; "Overhead" is the total time minus the max t_i (broadcasts, list of files, gather and merge).

| Run | Total time (s) | Max t_i (s) | Mean t_i (s) | Overhead (s) |
|---|---|---|---|---|
| 1 | 4.170 | 4.114 | 3.736 | 0.056 |
| 2 | 4.216 | 4.148 | 3.783 | 0.068 |
| 3 | 4.282 | 4.236 | 3.872 | 0.046 |
| 4 | 6.724 | 6.659 | 6.227 | 0.065 |
| 5 | 4.205 | 4.173 | 3.826 | 0.032 |

- **T_p = 4.720 s** (mean of 5, std 1.12 s), **S_p = 22.360 / 4.720 = 4.74**, **E_p = 4.74 / 11 = 0.43**.
- Median of the total times: 4.216 s, which gives S_p = 5.30 and E_p = 0.48.
- Run 4 is an outlier: all 11 processes were about 60% slower at the same time (the sum of their local times was 68.5 s instead of about 42 s), so something outside the program slowed down the whole machine during that run. It is kept in the mean, but without it T_p would be 4.218 s (std 0.05 s).
- All 5 runs printed `MATCHES_BASELINE=True` and the same top 10 as the baseline.

### c. Load imbalance evidence (version 1)

Local time t_i of every process (seconds) and LI = max(t_i) / mean(t_i):

| Run | r0 | r1 | r2 | r3 | r4 | r5 | r6 | r7 | r8 | r9 | r10 | LI | Spread |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 4.11 | 3.61 | 3.45 | 3.88 | 3.77 | 3.68 | 3.78 | 3.71 | 3.81 | 3.21 | 4.08 | 1.101 | 0.241 |
| 2 | 4.15 | 3.66 | 3.50 | 3.91 | 3.81 | 3.80 | 3.82 | 3.77 | 3.86 | 3.27 | 4.08 | 1.096 | 0.233 |
| 3 | 4.24 | 3.77 | 3.58 | 4.02 | 3.89 | 3.81 | 3.93 | 3.82 | 3.94 | 3.37 | 4.22 | 1.094 | 0.224 |
| 4 | 6.66 | 6.07 | 5.86 | 6.41 | 6.27 | 6.25 | 6.32 | 6.15 | 6.31 | 5.53 | 6.66 | 1.069 | 0.181 |
| 5 | 4.15 | 3.68 | 3.53 | 3.95 | 3.89 | 3.80 | 3.90 | 3.78 | 3.93 | 3.30 | 4.17 | 1.091 | 0.227 |

**Mean LI = 1.090**, mean relative spread = 0.221. The slowest process takes about 9% more than the average, and 22% more than the fastest one.

The reason is the amount of text, not the number of files. Every process received 272 or 273 files, but the words in its block were very different:

| Rank | r0 | r1 | r2 | r3 | r4 | r5 | r6 | r7 | r8 | r9 | r10 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Files | 273 | 273 | 273 | 273 | 273 | 273 | 273 | 273 | 272 | 272 | 272 |
| Tokens (millions) | 4.55 | 3.90 | 3.64 | 4.32 | 4.20 | 4.06 | 4.21 | 3.99 | 4.23 | 3.27 | 4.59 |

The order of the local times is the same as the order of the tokens in every run: rank 9 (3.27 M tokens) is always the fastest and ranks 0 and 10 (about 4.6 M) are always the slowest. The imbalance is repeatable and is caused by how many large files fell in each block. The total time follows the slowest process (max t_i), so with this distribution about 0.37 s of every run (mean of max t_i - mean t_i) is spent by the other processes waiting for the slowest ones.

### d. MPI version 2: how it corrects the imbalance, and its results

**Idea.** In version 1 every process gets the same number of files, but not the same amount of text. Version 2 keeps the same program (broadcast of the query, local counting with the same loop, gather in rank 0) and only changes how the files are given out:

1. Rank 0 lists the files and broadcasts the names (`comm.bcast`).
2. The size in bytes of a file is used as an estimate of its work (the counting time is proportional to the number of words). Reading the size of a file takes about 1 ms on the Docker volume of this machine (3000 files = about 3 s), so instead of doing it only in rank 0, each process reads the sizes of `files[rank::p]` and sends them to rank 0 with `comm.gather`.
3. Rank 0 sorts all files from the largest to the smallest and broadcasts the sorted list.
4. The sorted list is dealt like cards: process r takes positions r, r + p, r + 2p, ... (`sorted_files[rank::p]`). In the first rounds every process receives one of the large files, then one of the medium ones and at the end the small ones, so all of them end with about the same amount of text.

The assignment is still static (decided before counting starts). The time of steps 1-3 is printed as `SORT_TIME`; it is the extra cost of version 2 and it is included in the total time.

**Result of the assignment.** Every process got 272 or 273 files, as in version 1, but now between 4.04 M and 4.13 M tokens (a difference of 2%, against 40% in version 1: 3.27 M to 4.59 M).

| Run | Total time (s) | Sort time (s) | Max t_i (s) | Mean t_i (s) | LI | Spread |
|---|---|---|---|---|---|---|
| 1 | 4.303 | 0.503 | 3.789 | 3.740 | 1.013 | 0.027 |
| 2 | 4.351 | 0.561 | 3.777 | 3.715 | 1.017 | 0.034 |
| 3 | 4.995 | 0.736 | 4.246 | 4.206 | 1.010 | 0.020 |
| 4 | 4.710 | 0.842 | 3.837 | 3.783 | 1.014 | 0.025 |
| 5 | 4.302 | 0.509 | 3.782 | 3.756 | 1.007 | 0.016 |

- **T_p = 4.532 s** (mean of 5, std 0.31 s), **S_p = 22.360 / 4.532 = 4.93**, **E_p = 4.93 / 11 = 0.45**.
- Median of the total times: 4.351 s, which gives S_p = 5.14 and E_p = 0.47.
- **Mean LI = 1.012** and mean relative spread = 0.024. In every run the 11 local times are within 0.15 s of each other.
- Mean sort time = 0.630 s (0.50 s in the best runs, up to 0.84 s).
- All 5 runs printed `MATCHES_BASELINE=True` and the same top 10 as the baseline.

**Comparison (p = 11, T_seq = 22.360 s, ideal S_p = 11, ideal time = 2.03 s)**

| Metric | MPI v1 | MPI v2 |
|---|---|---|
| T_p, mean of 5 (s) | 4.720 | 4.532 |
| T_p, median (s) | 4.216 | 4.351 |
| Speedup S_p (mean / median) | 4.74 / 5.30 | 4.93 / 5.14 |
| Efficiency E_p (mean / median) | 0.43 / 0.48 | 0.45 / 0.47 |
| LI (mean) | 1.090 | 1.012 |
| Relative spread (mean) | 0.221 | 0.024 |
| Max t_i, mean (s) | 4.666 | 3.886 |
| Max t_i, without v1 run 4 (s) | 4.168 | 3.886 |
| Time outside counting (s) | 0.054 | 0.646 |

Version 2 clearly fixes the imbalance (LI from 1.090 to 1.012). The mean total time is better for version 2, but only because of the slow run 4 of version 1; with the median, version 1 is 0.135 s (3%) faster. Section 5.b explains why.

## 5. Analysis

*(pending)*

## 6. Conclusions

*(pending)*
