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
2. Run the sequential baseline 4 times and record its time and its output. T_seq is the mean.
3. Run `detectar_procesos_mpi.py` and use the recommended N as p for every MPI run.
4. Run `mpi1.py` with p processes at least 3 times. For each run record the total time and, for each process, the number of files and the local counting time t_i.
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
- **Performance:** the runs of each version are done one after another in the same session, with no other heavy programs open. The first baseline run is kept because it was not slower than the others.

## 4. Experimental Plan Execution

### a. Sequential baseline timing

| Run | 1 | 2 | 3 | 4 |
|---|---|---|---|---|
| Time (s) | 21.548 | 22.040 | 21.962 | 22.127 |

**T_seq = 21.919 s** (mean of the 4 runs, the difference between runs is below 3%). Output: 3000 files, 44,951,458 tokens, 3,631,778 occurrences.

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

*(pending)*

### c. Load imbalance evidence (version 1)

*(pending)*

### d. MPI version 2: how it corrects the imbalance, and its results

*(pending)*

## 5. Analysis

*(pending)*

## 6. Conclusions

*(pending)*
