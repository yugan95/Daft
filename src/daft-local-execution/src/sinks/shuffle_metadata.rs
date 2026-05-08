/// Summary statistics for a single output partition produced by a shuffle
/// writer (e.g. row count, serialised byte size).
///
/// Mirrors the pattern used by [`daft_partition_refs::FlightPartitionRef`]
/// which lives in its own dedicated module.
pub(crate) struct ShufflePartitionMeta {
    pub(crate) num_rows: usize,
    pub(crate) size_bytes: usize,
}

impl ShufflePartitionMeta {
    pub(crate) fn new(num_rows: usize, size_bytes: usize) -> Self {
        Self {
            num_rows,
            size_bytes,
        }
    }
}
