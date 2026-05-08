//! In-memory mock implementation of [`CelebornClient`].
//!
//! Used during Daft-side development before the real Celeborn Rust/FFI SDK is
//! available. All pushed bytes are kept in process memory keyed by
//! `(shuffle_id, partition_id)` so that subsequent `read_partition` calls
//! return them in push order.
//!
//! NOT suitable for production: data is lost when the process exits and there
//! is no fault tolerance, replication, or cross-process coordination.

use std::{
    collections::{HashMap, HashSet},
    sync::{Arc, Mutex},
};

use async_trait::async_trait;
use bytes::Bytes;
use common_error::DaftResult;
use futures::stream;

use crate::client::celeborn::client::{CelebornClient, PartitionDataStream};

/// Key identifying a single reduce partition within a shuffle.
type PartitionKey = (u64, u32);

#[derive(Default)]
struct MockState {
    /// All pushed bytes per partition, in arrival order.
    partitions: HashMap<PartitionKey, Vec<Bytes>>,
    /// Number of `mapper_end` calls per `(shuffle_id, map_id, attempt_id)`.
    /// Used by tests to assert mappers terminated correctly.
    mapper_ends: HashMap<(u64, u32, u32), u32>,
    /// Shuffles that have been unregistered.
    unregistered: HashSet<u64>,
    /// Per-shuffle metadata registered via `register_shuffle`.
    shuffle_meta: HashMap<u64, (u32, u32)>, // shuffle_id -> (num_mappers, num_partitions)
}

/// Process-local in-memory Celeborn client used for development and testing.
///
/// This is a **connection-level** mock: one instance can serve multiple
/// shuffles, just like the real [`ShuffleCelebornClient`](super::ffi::ShuffleCelebornClient).
pub struct MockShuffleCelebornClient {
    state: Arc<Mutex<MockState>>,
}

impl Default for MockShuffleCelebornClient {
    fn default() -> Self {
        Self::new()
    }
}

impl MockShuffleCelebornClient {
    pub fn new() -> Self {
        Self {
            state: Arc::new(Mutex::new(MockState::default())),
        }
    }

    /// Inspect how many `mapper_end` calls were made for the given map attempt.
    /// Useful for unit tests.
    pub fn mapper_end_count(&self, shuffle_id: u64, map_id: u32, attempt_id: u32) -> u32 {
        let state = self.state.lock().expect("mock state poisoned");
        state
            .mapper_ends
            .get(&(shuffle_id, map_id, attempt_id))
            .copied()
            .unwrap_or(0)
    }

    /// Inspect how many partition blocks were pushed for the given partition.
    pub fn pushed_block_count(&self, shuffle_id: u64, partition_id: u32) -> usize {
        let state = self.state.lock().expect("mock state poisoned");
        state
            .partitions
            .get(&(shuffle_id, partition_id))
            .map(Vec::len)
            .unwrap_or(0)
    }

    /// Whether the given shuffle has been unregistered.
    pub fn is_unregistered(&self, shuffle_id: u64) -> bool {
        let state = self.state.lock().expect("mock state poisoned");
        state.unregistered.contains(&shuffle_id)
    }
}

#[async_trait]
impl CelebornClient for MockShuffleCelebornClient {
    async fn register_shuffle(
        &self,
        shuffle_id: u64,
        num_mappers: u32,
        num_partitions: u32,
    ) -> DaftResult<()> {
        let mut state = self.state.lock().expect("mock state poisoned");
        state
            .shuffle_meta
            .insert(shuffle_id, (num_mappers, num_partitions));
        Ok(())
    }

    async fn push_data(
        &self,
        shuffle_id: u64,
        _map_id: u32,
        _attempt_id: u32,
        partition_id: u32,
        data: &[u8],
    ) -> DaftResult<()> {
        // NOTE: `map_id` and `attempt_id` are intentionally ignored in the
        // mock because real Celeborn aggregates all mappers' data for the
        // same partition into a single stream. Keying only on
        // `(shuffle_id, partition_id)` faithfully mirrors that behaviour.
        let mut state = self.state.lock().expect("mock state poisoned");
        state
            .partitions
            .entry((shuffle_id, partition_id))
            .or_default()
            .push(Bytes::copy_from_slice(data));
        Ok(())
    }

    async fn mapper_end(&self, shuffle_id: u64, map_id: u32, attempt_id: u32) -> DaftResult<()> {
        let mut state = self.state.lock().expect("mock state poisoned");
        *state
            .mapper_ends
            .entry((shuffle_id, map_id, attempt_id))
            .or_insert(0) += 1;
        Ok(())
    }

    async fn read_partition(
        &self,
        shuffle_id: u64,
        partition_id: u32,
    ) -> DaftResult<PartitionDataStream> {
        let chunks = {
            let state = self.state.lock().expect("mock state poisoned");
            state
                .partitions
                .get(&(shuffle_id, partition_id))
                .cloned()
                .unwrap_or_default()
        };
        Ok(Box::pin(stream::iter(chunks.into_iter().map(Ok))))
    }

    async fn unregister_shuffle(&self, shuffle_id: u64) -> DaftResult<()> {
        let mut state = self.state.lock().expect("mock state poisoned");
        state.unregistered.insert(shuffle_id);
        state.partitions.retain(|(sid, _), _| *sid != shuffle_id);
        state.shuffle_meta.remove(&shuffle_id);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use daft_core::{
        datatypes::{Float64Array, Int32Array, Utf8Array},
        series::IntoSeries,
    };
    use daft_micropartition::MicroPartition;
    use daft_recordbatch::RecordBatch;
    use futures::StreamExt;

    use super::*;

    fn new_client() -> MockShuffleCelebornClient {
        MockShuffleCelebornClient::new()
    }

    #[tokio::test]
    async fn push_then_read_returns_data_in_order() -> DaftResult<()> {
        let client = new_client();
        client.register_shuffle(1, 2, 8).await?;
        client.push_data(1, 0, 0, 7, b"hello").await?;
        client.push_data(1, 1, 0, 7, b"world").await?;

        let mut stream = client.read_partition(1, 7).await?;
        let mut collected = Vec::new();
        while let Some(chunk) = stream.next().await {
            collected.push(chunk?);
        }
        assert_eq!(collected.len(), 2);
        assert_eq!(collected[0].as_ref(), b"hello");
        assert_eq!(collected[1].as_ref(), b"world");
        Ok(())
    }

    #[tokio::test]
    async fn mapper_end_is_recorded() -> DaftResult<()> {
        let client = new_client();
        client.register_shuffle(42, 10, 1).await?;
        client.mapper_end(42, 3, 0).await?;
        assert_eq!(client.mapper_end_count(42, 3, 0), 1);
        assert_eq!(client.mapper_end_count(42, 3, 1), 0);
        Ok(())
    }

    #[tokio::test]
    async fn unregister_drops_data_and_marks_flag() -> DaftResult<()> {
        let client = new_client();
        client.register_shuffle(9, 1, 1).await?;
        client.push_data(9, 0, 0, 0, b"x").await?;
        assert_eq!(client.pushed_block_count(9, 0), 1);

        client.unregister_shuffle(9).await?;
        assert!(client.is_unregistered(9));
        assert_eq!(client.pushed_block_count(9, 0), 0);
        Ok(())
    }

    #[tokio::test]
    async fn read_unknown_partition_returns_empty_stream() -> DaftResult<()> {
        let client = new_client();
        client.register_shuffle(123, 1, 457).await?;
        let mut stream = client.read_partition(123, 456).await?;
        assert!(stream.next().await.is_none());
        Ok(())
    }

    /// Two mappers each push one block to the same partition. `read_partition`
    /// must return both blocks, in push order. This mirrors the runtime
    /// behavior expected by `RepartitionSink` (each mapper calls `push_data`
    /// once per non-empty partition) and by `CelebornShuffleReadSource`
    /// (the reducer expects all blocks for a partition to arrive on a single
    /// stream).
    #[tokio::test]
    async fn multiple_mappers_pushed_to_same_partition_are_concatenated() -> DaftResult<()> {
        let client = new_client();
        client.register_shuffle(1, 3, 6).await?;

        client
            .push_data(1, /* map_id */ 0, 0, 5, b"from_mapper_0")
            .await?;
        client
            .push_data(1, /* map_id */ 1, 0, 5, b"from_mapper_1")
            .await?;
        client
            .push_data(1, /* map_id */ 2, 0, 5, b"from_mapper_2")
            .await?;

        client.mapper_end(1, 0, 0).await?;
        client.mapper_end(1, 1, 0).await?;
        client.mapper_end(1, 2, 0).await?;

        // All three mappers must have recorded exactly one mapper_end call.
        for map_id in 0..3 {
            assert_eq!(
                client.mapper_end_count(1, map_id, 0),
                1,
                "mapper {map_id} should have exactly one mapper_end call"
            );
        }

        // Reducer reads partition 5 and gets all three blocks.
        let mut stream = client.read_partition(1, 5).await?;
        let mut chunks = Vec::new();
        while let Some(chunk) = stream.next().await {
            chunks.push(chunk?);
        }
        assert_eq!(chunks.len(), 3);
        assert_eq!(chunks[0].as_ref(), b"from_mapper_0");
        assert_eq!(chunks[1].as_ref(), b"from_mapper_1");
        assert_eq!(chunks[2].as_ref(), b"from_mapper_2");
        Ok(())
    }

    /// Different shuffles must not see each other's data, even when they
    /// reuse the same `(map_id, partition_id)` coordinates.
    #[tokio::test]
    async fn shuffles_are_isolated_by_shuffle_id() -> DaftResult<()> {
        let client = new_client();

        client.register_shuffle(10, 1, 1).await?;
        client.register_shuffle(20, 1, 1).await?;

        client.push_data(10, 0, 0, 0, b"shuffle_10_data").await?;
        client.push_data(20, 0, 0, 0, b"shuffle_20_data").await?;

        let mut s10 = client.read_partition(10, 0).await?;
        let chunk10 = s10.next().await.unwrap()?;
        assert_eq!(chunk10.as_ref(), b"shuffle_10_data");
        assert!(s10.next().await.is_none());

        let mut s20 = client.read_partition(20, 0).await?;
        let chunk20 = s20.next().await.unwrap()?;
        assert_eq!(chunk20.as_ref(), b"shuffle_20_data");
        assert!(s20.next().await.is_none());

        // Unregistering shuffle 10 must not affect shuffle 20.
        client.unregister_shuffle(10).await?;
        assert!(client.is_unregistered(10));
        assert!(!client.is_unregistered(20));
        assert_eq!(client.pushed_block_count(10, 0), 0);
        assert_eq!(client.pushed_block_count(20, 0), 1);
        Ok(())
    }

    /// `unregister_shuffle` must be idempotent: invoking it twice should leave
    /// the client in the same state and not produce an error.
    #[tokio::test]
    async fn unregister_is_idempotent() -> DaftResult<()> {
        let client = new_client();
        client.register_shuffle(7, 1, 1).await?;
        client.push_data(7, 0, 0, 0, b"data").await?;

        client.unregister_shuffle(7).await?;
        client.unregister_shuffle(7).await?; // second call must not error or duplicate flag.

        assert!(client.is_unregistered(7));
        // Internal state still records the shuffle exactly once. We verify by
        // unregistering a fresh shuffle and ensuring `is_unregistered(7)` is
        // unchanged.
        client.unregister_shuffle(99).await?;
        assert!(client.is_unregistered(7));
        assert!(client.is_unregistered(99));
        Ok(())
    }

    /// End-to-end data path test: construct a real `MicroPartition`, serialize
    /// it to Arrow IPC bytes (exactly as `RepartitionSink::sink` does), push
    /// the bytes through `MockShuffleCelebornClient`, read them back via
    /// `read_partition`, deserialize back to `MicroPartition`, and assert that
    /// the data content is identical.
    ///
    /// This is the **most important test** in the Celeborn integration because
    /// it exercises the same serialization pipeline that production will use:
    ///
    /// ```text
    ///   MicroPartition
    ///     → write_to_ipc_stream() → Vec<u8>
    ///       → push_data(bytes) → [MockShuffleCelebornClient stores]
    ///         → read_partition() → Stream<Bytes>
    ///           → read_from_ipc_stream(bytes) → MicroPartition
    /// ```
    #[tokio::test]
    async fn arrow_ipc_roundtrip_through_push_and_read() -> DaftResult<()> {
        let client = new_client();

        // Build a MicroPartition with mixed types (int, float, string).
        let string_values = vec!["alpha", "beta", "gamma"];
        let batch = RecordBatch::from_nonempty_columns(vec![
            Int32Array::from_slice("id", &[10, 20, 30]).into_series(),
            Float64Array::from_slice("score", &[1.5, 2.5, 3.5]).into_series(),
            Utf8Array::from_slice("name", string_values.as_slice()).into_series(),
        ])?;

        let original =
            MicroPartition::new_loaded(batch.schema.clone(), Arc::new(vec![batch.clone()]), None);

        // Serialize to IPC (exactly as RepartitionSink Celeborn branch does).
        let ipc_bytes = original.write_to_ipc_stream()?;

        // Push through the mock client.
        let shuffle_id = 100;
        let partition_id = 7;
        client.register_shuffle(100, 1, 8).await?;
        client
            .push_data(shuffle_id, 0, 0, partition_id, &ipc_bytes)
            .await?;

        // Read back.
        let mut stream = client.read_partition(shuffle_id, partition_id).await?;
        let chunk = stream.next().await.expect("should have one chunk")?;
        assert!(
            stream.next().await.is_none(),
            "should have exactly one chunk"
        );

        // Deserialize (exactly as CelebornShuffleReadSource does).
        let roundtrip = MicroPartition::read_from_ipc_stream(&chunk)?;

        // Assert schema matches.
        assert_eq!(original.schema(), roundtrip.schema());

        // Assert row count.
        assert_eq!(roundtrip.len(), 3);

        // Assert data content is identical at the RecordBatch level.
        assert_eq!(roundtrip.record_batches().len(), 1);
        assert_eq!(batch, roundtrip.record_batches()[0]);
        Ok(())
    }

    /// Two mappers each push an IPC-serialized `MicroPartition` to the same
    /// partition. The reducer reads them back as two separate chunks and
    /// deserializes each independently. This mirrors the real multi-mapper
    /// shuffle topology where `CelebornShuffleReadSource` calls
    /// `read_from_ipc_stream` on each chunk in the stream.
    #[tokio::test]
    async fn multi_mapper_arrow_ipc_roundtrip() -> DaftResult<()> {
        let client = new_client();
        let shuffle_id = 200;
        let partition_id = 0;

        client.register_shuffle(shuffle_id, 2, 1).await?;

        // Mapper 0 pushes 2 rows.
        let batch_m0 = RecordBatch::from_nonempty_columns(vec![
            Int32Array::from_slice("x", &[1, 2]).into_series(),
        ])?;
        let mp_m0 = MicroPartition::new_loaded(
            batch_m0.schema.clone(),
            Arc::new(vec![batch_m0.clone()]),
            None,
        );
        let ipc_m0 = mp_m0.write_to_ipc_stream()?;
        client
            .push_data(shuffle_id, 0, 0, partition_id, &ipc_m0)
            .await?;

        // Mapper 1 pushes 3 rows (same schema).
        let batch_m1 = RecordBatch::from_nonempty_columns(vec![
            Int32Array::from_slice("x", &[3, 4, 5]).into_series(),
        ])?;
        let mp_m1 = MicroPartition::new_loaded(
            batch_m1.schema.clone(),
            Arc::new(vec![batch_m1.clone()]),
            None,
        );
        let ipc_m1 = mp_m1.write_to_ipc_stream()?;
        client
            .push_data(shuffle_id, 1, 0, partition_id, &ipc_m1)
            .await?;

        // Read back: should get 2 chunks.
        let mut stream = client.read_partition(shuffle_id, partition_id).await?;

        let chunk0 = stream.next().await.expect("chunk from mapper 0")?;
        let rt0 = MicroPartition::read_from_ipc_stream(&chunk0)?;
        assert_eq!(rt0.len(), 2);
        assert_eq!(batch_m0, rt0.record_batches()[0]);

        let chunk1 = stream.next().await.expect("chunk from mapper 1")?;
        let rt1 = MicroPartition::read_from_ipc_stream(&chunk1)?;
        assert_eq!(rt1.len(), 3);
        assert_eq!(batch_m1, rt1.record_batches()[0]);

        assert!(stream.next().await.is_none(), "no more chunks");

        // Total rows across both chunks: 2 + 3 = 5.
        let total_rows: usize = rt0.len() + rt1.len();
        assert_eq!(total_rows, 5);
        Ok(())
    }
}
