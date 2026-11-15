# 1. Parallel Word Counting in a Text Corpus with MPI

**Author:** Sebastián Mercado M. - Estructura del Computador II, Universidad del Norte.

## 2. Problem Description

`generator.py` (seed 42) creates the folder `dataset/` with `consulta.txt` (200 query words, one per line) and 3000 text files `file_XXXX.txt` with 44,951,458 words in total. The files have very different sizes: about 60% small, 30% medium and 10% large.

The goal is to count how many times each query word appears in the corpus and report the 10 most frequent ones. The problem is solved with the sequential baseline and with two MPI programs, and all of them must give the same counts.

| File | Description |
|---|---|
| `generator.py`, `baseline_secuencial.py`, `run_all.sh` | Provided by the instructor, not modified |
| `detectar_procesos_mpi.py` | Provided script used to choose the number of processes p |
| `mpi1.py` | MPI v1: broadcast + static distribution in blocks of files |
| `mpi2.py` | MPI v2: files sorted by size and dealt like cards |
| `evidence/` | Raw output of every run used in this report |

## 3. Environment and Execution Instructions

Laptop with an Intel Core i5-12450H (8 cores: 4 performance cores with 2 threads each + 4 efficiency cores, 12 threads), 32 GB RAM, Windows 11 and Docker Desktop. Everything runs in the image `augustosalazar/slim-mpi:2` (Python 3.8, mpi4py 4.1.1, Open MPI 4.1.4). Commands in PowerShell, from the repository folder:

```powershell
docker run --rm -v "${PWD}:/app" augustosalazar/slim-mpi:2 python /app/generator.py
docker run --rm -v "${PWD}:/app" augustosalazar/slim-mpi:2 python /app/baseline_secuencial.py
docker run --rm -v "${PWD}:/app" augustosalazar/slim-mpi:2 python /app/detectar_procesos_mpi.py
docker run --rm -v "${PWD}:/app" augustosalazar/slim-mpi:2 mpirun --allow-run-as-root --oversubscribe -np 11 python /app/mpi1.py
docker run --rm -v "${PWD}:/app" augustosalazar/slim-mpi:2 mpirun --allow-run-as-root --oversubscribe -np 11 python /app/mpi2.py
docker run --rm -e NP_LIST=11 -e RUNS=3 -v "${PWD}:/app" augustosalazar/slim-mpi:2 sh /app/run_all.sh
```

The file `.gitattributes` keeps `run_all.sh` with Linux line endings so it works in the container when the repository is cloned on Windows.

### a. Experimental Plan

1. Generate the dataset.
2. Run the baseline 6 times (4 before and 2 after the MPI runs). T_seq is the mean.
3. Choose p with `detectar_procesos_mpi.py`.
4. Run `mpi1.py` and `mpi2.py` 5 times each with p processes, recording the total time and the files and local time t_i of every process.
5. Compute T_p (mean total time), S_p = T_seq / T_p, E_p = S_p / p, LI = max(t_i) / mean(t_i) and the relative spread (max(t_i) - min(t_i)) / mean(t_i).
6. Compare both versions.

### b. Sequential baseline

`baseline_secuencial.py` reads the query words into a set, reads every file line by line, converts each word to lower case and adds 1 to a `Counter` when the word is in the set. It prints the time, the top 10, and saves all counts in `dataset/baseline_results.csv`.

### c. MPI version 1

1. Rank 0 reads `consulta.txt` and sends the words to all processes with `comm.bcast`.
2. Rank 0 lists the `file_*.txt` files and sends the list with `comm.bcast`.
3. Static distribution: the list is split into p blocks with the same number of files (272 or 273 for p = 11), and process r takes block r.
4. Each process counts the words in its files (same loop as the baseline) and measures its local time with `MPI.Wtime()`.
5. `comm.gather` sends the partial counts, number of files and local time of every process to rank 0.
6. Rank 0 adds the counts and prints the report of every process, the LI, the top 10, `MATCHES_BASELINE` and `EXECUTION_TIME` (the line read by `run_all.sh`).

The total time starts after `comm.Barrier()` and ends when rank 0 has the final result.

### d. Test procedure

- **Correctness:** every MPI run must print `MATCHES_BASELINE=True`, which means its full table of counts is equal to `baseline_results.csv`.
- **Performance:** all runs were made in the same session with no other heavy programs open. The MPI runs were interleaved (v1, v2, v1, v2, ...) so that changes in the machine affect both versions equally.
- **Final check:** `run_all.sh` was run once at the end with `NP_LIST=11` and `RUNS=3`.

## 4. Experimental Plan Execution

### a. Sequential baseline timing

| Run | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|
| Time (s) | 21.548 | 22.040 | 21.962 | 22.127 | 24.277 | 22.205 |

**T_seq = 22.36 s.** Output: 3000 files, 44,951,458 tokens, 3,631,778 occurrences. Top 10: a 785774, para 392156, sus 228913, otros 105530, ante 99832, unos 88794, otra 83901, vosotros 61617, mios 58420, tuya 56635 (same as the instructor's example).

**Choice of p.** `detectar_procesos_mpi.py` tested N = 1 to 12:

| N | 1 | 2 | 4 | 6 | 8 | 10 | 11 | 12 |
|---|---|---|---|---|---|---|---|---|
| Median (s) | 3.685 | 2.233 | 1.618 | 1.670 | 1.211 | 1.311 | **1.180** | 1.921 |

It recommended N = 11, so **p = 11**. N = 8, 10 and 11 were very close.

### b. MPI version 1 timing results

| Run | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|
| Total time (s) | 4.170 | 4.216 | 4.282 | 6.724 | 4.205 |

**T_p = 4.720 s, S_p = 4.74, E_p = 0.43.** Run 4 was slow in all processes at the same time (something outside the program); without it T_p = 4.218 s. All runs printed `MATCHES_BASELINE=True`.

### c. Load imbalance evidence (version 1)

Local time of every process in run 1 (the other runs show the same pattern, see `evidence/official_runs/`):

| Rank | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Files | 273 | 273 | 273 | 273 | 273 | 273 | 273 | 273 | 272 | 272 | 272 |
| Tokens (M) | 4.55 | 3.90 | 3.64 | 4.32 | 4.20 | 4.06 | 4.21 | 3.99 | 4.23 | 3.27 | 4.59 |
| t_i (s) | 4.11 | 3.61 | 3.45 | 3.88 | 3.77 | 3.68 | 3.78 | 3.71 | 3.81 | 3.21 | 4.08 |

LI per run: 1.101, 1.096, 1.094, 1.069, 1.091. **Mean LI = 1.090**, mean spread = 0.22.

All processes have the same number of files but not the same amount of text, and the local time follows the tokens: rank 9 (3.27 M) is the fastest and ranks 0 and 10 (4.6 M) are the slowest in all runs.

### d. MPI version 2: how it corrects the imbalance, and its results

Version 2 is the same program as version 1 but changes how the files are given out:

1. Each process reads the size of part of the files and sends them to rank 0 with `comm.gather` (reading the size of one file takes about 1 ms in the Docker volume, so it is split among all processes).
2. Rank 0 sorts the files from largest to smallest and sends the list with `comm.bcast`.
3. The sorted list is dealt like cards: process r takes files r, r + p, r + 2p, ... In each round every process gets a file of similar size, so all end with about the same amount of text.

The time of these steps is printed as `SORT_TIME` and is part of the total time. Now every process gets between 4.04 M and 4.13 M tokens (2% difference, against 40% in v1).

| Run | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|
| Total time (s) | 4.303 | 4.351 | 4.995 | 4.710 | 4.302 |
| Sort time (s) | 0.503 | 0.561 | 0.736 | 0.842 | 0.509 |
| LI | 1.013 | 1.017 | 1.010 | 1.014 | 1.007 |

**T_p = 4.532 s, S_p = 4.93, E_p = 0.45, mean LI = 1.012.** All runs printed `MATCHES_BASELINE=True`.

**Comparison (p = 11, T_seq = 22.36 s, ideal S_p = 11)**

| | MPI v1 | MPI v2 |
|---|---|---|
| T_p mean (s) | 4.720 | 4.532 |
| T_p median (s) | 4.216 | 4.351 |
| S_p (mean / median) | 4.74 / 5.30 | 4.93 / 5.14 |
| E_p (mean / median) | 0.43 / 0.48 | 0.45 / 0.47 |
| LI | 1.090 | 1.012 |
| Slowest process, normal runs (s) | 4.17 | 3.89 |
| `run_all.sh` check, mean of 3 (s) | 4.733 | 4.494 |

The 6 runs of `run_all.sh` also printed `MATCHES_BASELINE=True`.

## 5. Analysis

### a. Parallel performance

Version 1 improved the baseline from 22.36 s to 4.72 s (4.22 s median), a speedup of 4.7-5.3. This is far from the ideal of 11, with an efficiency below 0.5.

The gap is not communication: the total time is only about 0.05 s larger than the slowest process. The imbalance explains about 0.37 s. The main reason is that each process works slower when 11 run together: the local times add up to about 41 s, almost twice the 22.36 s of the sequential program. The laptop has only 8 physical cores (4 of them slower) for 11 processes, and all of them read files from the same disk through Docker. The calibration script showed the same limit: its best N was only 3.1 times faster than N = 1.

### b. Load balance

Version 1 showed a clear and repeatable imbalance: LI = 1.090, and the processes with more large files (more tokens) were always the slowest. Version 2 removed it: LI = 1.012 and the slowest process went from 4.17 s to 3.89 s. This happened in all runs (v1: LI 1.069-1.101, v2: LI 1.007-1.019).

But this did not give a clear improvement in total time. Balancing can save at most about 0.3 s, and sorting the files costs about 0.6 s, mostly from reading the 3000 file sizes. Because of this, the winner changes with the measure: v2 is faster in the mean and in `run_all.sh`, v1 is faster in the median. The differences (3-5%) are about the same size as the variation between runs.

### c. Limitations

- **Docker on Windows:** reading file sizes is slow in this setup, and that is the extra cost of v2. On another machine the result between v1 and v2 could be different.
- **Noise:** one v1 run was 60% slower and one baseline run 9% slower, so differences of a few percent between the versions are not reliable with 5 runs. The LI results did not overlap in any run, so that conclusion is solid.
- **Hardware:** 11 processes on 8 physical cores of two types limits the speedup, so the low efficiency describes this laptop more than the programs.
- **Choice of p:** the calibration measures a different task, the best values were very close, and only one p was tested.

These limitations mainly affect the comparison of total times between v1 and v2. They do not change that both versions are about 5 times faster than the baseline and that v2 removes the imbalance.

## 6. Conclusions

I would recommend the strategy of version 2 (files sorted by size and dealt like cards). It was the only change that improved the distribution in every run: the LI went from 1.090 to 1.012 and the slowest process went from 4.17 s to 3.89 s, because every process received almost the same amount of text (4.04-4.13 M tokens against 3.27-4.59 M in version 1). In total time it reached T_p = 4.53 s, S_p = 4.93 and E_p = 0.45, against 4.72 s, 4.74 and 0.43 for version 1, and it was also faster in the `run_all.sh` check (4.49 s against 4.73 s). However, the advantage is small and not certain: with the median, version 1 was 3% faster (S_p = 5.30 against 5.14), because in this Docker setup reading the 3000 file sizes costs about 0.6 s, almost all of what the balance saves. So the recommendation is moderate: version 2 balances the work much better and is at least as fast, and it should be more useful when the file sizes are cheaper to read or the imbalance is larger. In both cases the speedup stays far from the ideal of 11, mainly because of the hardware, not the distribution strategy.
