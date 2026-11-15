# MPI version 1: broadcast of the query + static distribution of files.
#
# 1. rank 0 reads consulta.txt
# 2. rank 0 broadcasts the query words (comm.bcast)
# 3. rank 0 lists the file_*.txt files and broadcasts the list
# 4. static distribution: the list is split into p contiguous blocks with
#    the same number of files (+-1); process r takes block r
# 5. each process counts the query words in its own files
# 6. partial results are gathered in rank 0 (comm.gather)
# 7. rank 0 merges them and prints the per-process report and the top 10
#
# Run (inside the container):
#   mpirun --allow-run-as-root --oversubscribe -np 4 python /app/mpi1.py

import os
from collections import Counter

from mpi4py import MPI

CONSULTA_NAME = "consulta.txt"
CASE_SENSITIVE = False
TOP_N = 10


def load_query(path):
    # same rules as the baseline: one word per line, lower case
    with open(path, "r", encoding="utf-8") as f:
        words = [line.strip() for line in f if line.strip()]
    if not CASE_SENSITIVE:
        words = [w.lower() for w in words]
    return set(words)


def list_files(dataset_dir):
    names = sorted(n for n in os.listdir(dataset_dir)
                   if n.startswith("file_") and n.endswith(".txt"))
    return [os.path.join(dataset_dir, n) for n in names]


def split_blocks(items, parts):
    # contiguous blocks, sizes differ by at most one file
    base, extra = divmod(len(items), parts)
    blocks = []
    start = 0
    for i in range(parts):
        end = start + base + (1 if i < extra else 0)
        blocks.append(items[start:end])
        start = end
    return blocks


def count_words(paths, targets):
    # same counting loop as baseline_secuencial.py
    counts = Counter()
    tokens = 0
    for path in paths:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                words = line.split()
                if not CASE_SENSITIVE:
                    words = [w.lower() for w in words]
                tokens += len(words)
                for w in words:
                    if w in targets:
                        counts[w] += 1
    return counts, tokens


def check_baseline(counts, csv_path):
    # True/False if the full table equals baseline_results.csv, None if missing
    if not os.path.isfile(csv_path):
        return None
    expected = {}
    with open(csv_path, "r", encoding="utf-8") as f:
        next(f)
        for line in f:
            word, c = line.strip().rsplit(",", 1)
            expected[word] = int(c)
    return expected == dict(counts)


def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    dataset_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset")

    comm.Barrier()              # everybody starts the clock together
    t_start = MPI.Wtime()

    # steps 1-2: query read by rank 0 and broadcast
    query = None
    if rank == 0:
        query = load_query(os.path.join(dataset_dir, CONSULTA_NAME))
    query = comm.bcast(query, root=0)

    # steps 3-4: rank 0 lists the files and broadcasts the list,
    # then each process takes its own contiguous block using its rank
    files = None
    if rank == 0:
        files = list_files(dataset_dir)
    files = comm.bcast(files, root=0)
    my_files = split_blocks(files, size)[rank]

    # step 5: local counting, timed on its own
    t0 = MPI.Wtime()
    counts, tokens = count_words(my_files, query)
    local_time = MPI.Wtime() - t0

    # step 6: partial results to rank 0
    info = (rank, len(my_files), tokens, local_time, dict(counts))
    gathered = comm.gather(info, root=0)

    # step 7: global result
    if rank == 0:
        total = Counter()
        for part in gathered:
            total.update(part[4])
        elapsed = MPI.Wtime() - t_start

        times = [g[3] for g in gathered]
        mean_t = sum(times) / len(times)
        li = max(times) / mean_t
        spread = (max(times) - min(times)) / mean_t

        print(f"MPI version 1 (static contiguous blocks), p = {size}")
        print(f"Files processed: {sum(g[1] for g in gathered)}")
        print(f"Tokens read: {sum(g[2] for g in gathered)}")
        print(f"Total occurrences found: {sum(total.values())}\n")

        print("Per-process report:")
        for r, n_files, n_tokens, t, _ in gathered:
            print(f"  rank {r}: files={n_files} tokens={n_tokens} local_time={t:.6f} s")

        print(f"\nLOCAL_TIMES={','.join(f'{t:.6f}' for t in times)}")
        print(f"LOAD_IMBALANCE_LI={li:.4f}")
        print(f"RELATIVE_SPREAD={spread:.4f}")
        print(f"MATCHES_BASELINE={check_baseline(total, os.path.join(dataset_dir, 'baseline_results.csv'))}\n")

        print(f"Top {TOP_N} query words in the corpus:")
        for word, c in total.most_common(TOP_N):
            print(f"  {word}: {c}")

        print(f"\nEXECUTION_TIME={elapsed:.6f}")


if __name__ == "__main__":
    main()
