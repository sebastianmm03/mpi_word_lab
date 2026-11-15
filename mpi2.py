# MPI version 2: same program as version 1, but the files are dealt by size.
#
# Problem seen in version 1: every process gets the same NUMBER of files,
# but the files have very different sizes, so some processes read much more
# text than others and the slowest one decides the total time.
#
# Fix: rank 0 sorts the files from largest to smallest (size in bytes) and
# the sorted list is dealt like cards: process r takes positions
# r, r + p, r + 2p, ...  In every "round" each process gets one file of
# similar size, so all of them end with about the same amount of text.
#
# 1. rank 0 reads consulta.txt and broadcasts the words (comm.bcast)
# 2. rank 0 lists the files and broadcasts the names; each process reads
#    the sizes of a part of them; rank 0 gathers the sizes, sorts the files
#    from largest to smallest and broadcasts the sorted list
# 3. process r takes sorted_files[r::p]
# 4. local counting, gather in rank 0, top 10 (same as version 1)
#
# Run (inside the container):
#   mpirun --allow-run-as-root --oversubscribe -np 4 python /app/mpi2.py

import os
from collections import Counter

from mpi4py import MPI

CONSULTA_NAME = "consulta.txt"
CASE_SENSITIVE = False
TOP_N = 10


def load_query(path):
    with open(path, "r", encoding="utf-8") as f:
        words = [line.strip() for line in f if line.strip()]
    if not CASE_SENSITIVE:
        words = [w.lower() for w in words]
    return set(words)


def list_files(dataset_dir):
    names = sorted(n for n in os.listdir(dataset_dir)
                   if n.startswith("file_") and n.endswith(".txt"))
    return [os.path.join(dataset_dir, n) for n in names]


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

    comm.Barrier()
    t_start = MPI.Wtime()

    # 1. query read by rank 0 and broadcast
    query = None
    if rank == 0:
        query = load_query(os.path.join(dataset_dir, CONSULTA_NAME))
    query = comm.bcast(query, root=0)

    # 2. sorting phase (timed, it is the extra cost of version 2)
    t_sort = MPI.Wtime()

    # 2a. rank 0 lists the files and broadcasts the names
    files = None
    if rank == 0:
        files = list_files(dataset_dir)
    files = comm.bcast(files, root=0)

    # 2b. reading the size of a file is slow on the Docker volume (~1 ms),
    #     so every process reads the sizes of a part of the files (dealt
    #     like cards) and sends them to rank 0
    my_sizes = [(os.path.getsize(p), p) for p in files[rank::size]]
    all_sizes = comm.gather(my_sizes, root=0)

    # 2c. rank 0 sorts all files from largest to smallest and broadcasts them
    sorted_files = None
    if rank == 0:
        sorted_files = []
        for part in all_sizes:
            sorted_files += part
        sorted_files.sort(reverse=True)
    sorted_files = comm.bcast(sorted_files, root=0)
    sort_time = MPI.Wtime() - t_sort

    # 3. deal the sorted list like cards: process r takes r, r+p, r+2p, ...
    mine = sorted_files[rank::size]
    my_files = [p for s, p in mine]
    my_bytes = sum(s for s, p in mine)

    # 4. local counting
    t0 = MPI.Wtime()
    counts, tokens = count_words(my_files, query)
    local_time = MPI.Wtime() - t0

    info = (rank, len(my_files), my_bytes, tokens, local_time, dict(counts))
    gathered = comm.gather(info, root=0)

    if rank == 0:
        total = Counter()
        for part in gathered:
            total.update(part[5])
        elapsed = MPI.Wtime() - t_start

        times = [g[4] for g in gathered]
        mean_t = sum(times) / len(times)
        li = max(times) / mean_t
        spread = (max(times) - min(times)) / mean_t

        print(f"MPI version 2 (files sorted by size and dealt like cards), p = {size}")
        print(f"Files processed: {sum(g[1] for g in gathered)}")
        print(f"Tokens read: {sum(g[3] for g in gathered)}")
        print(f"Total occurrences found: {sum(total.values())}\n")

        print("Per-process report:")
        for r, n_files, n_bytes, n_tokens, t, _ in gathered:
            print(f"  rank {r}: files={n_files} bytes={n_bytes} tokens={n_tokens} local_time={t:.6f} s")

        print(f"\nSORT_TIME={sort_time:.6f}")
        print(f"LOCAL_TIMES={','.join(f'{t:.6f}' for t in times)}")
        print(f"LOAD_IMBALANCE_LI={li:.4f}")
        print(f"RELATIVE_SPREAD={spread:.4f}")
        print(f"MATCHES_BASELINE={check_baseline(total, os.path.join(dataset_dir, 'baseline_results.csv'))}\n")

        print(f"Top {TOP_N} query words in the corpus:")
        for word, c in total.most_common(TOP_N):
            print(f"  {word}: {c}")

        print(f"\nEXECUTION_TIME={elapsed:.6f}")


if __name__ == "__main__":
    main()
