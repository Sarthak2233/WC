# DSA Optimization Policy

## Algorithmic Efficiency
- **Complexity Goal:** Aim for $O(N \log N)$ or $O(N)$ for data processing. Avoid $O(N^2)$ on large datasets (e.g., player bio scraping results from **11 Data Layers**).
- **Graph Analysis:** When modeling player-club-country networks, use optimized libraries like `networkx` or `igraph`.
- **Search & Join:** Use hash-based joins (Pandas `merge`) or binary search for lookups in sorted data. Converge into **7 Master Tables** efficiently.

## Memory Management
- **Lazy Loading:** Use generators for processing large bio archives or news datasets (Layer 11) to avoid OOM errors.
- **Partitioning:** Partition historical datasets (1930-2022) by tournament year to limit memory pressure during ETL.
- **Data Types:** Downcast numeric columns in DataFrames whenever possible.
- **Parallelism:** Utilize `multiprocessing` or `concurrent.futures` for CPU-bound tasks like NLP feature extraction from player biographies.

## Code Patterns
- **Memoization:** Use `functools.lru_cache` for expensive calculations (e.g., historical tournament lookups).
- **Batching:** Process API calls and database writes in batches.

## Verification
- Every PRD/Plan must include a "DSA Impact" section detailing the expected time/space complexity of the proposed solution.
