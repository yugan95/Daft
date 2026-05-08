//! Celeborn shuffle client abstraction.
//!
//! This module defines the [`CelebornClient`] trait that abstracts away the
//! concrete Celeborn SDK implementation. The Daft shuffle layer (`RepartitionSink`,
//! `ShuffleReadSource`) depends only on this trait so that:
//!
//! * Daft side development can proceed independently of the upstream Celeborn
//!   client SDK availability.
//! * The mock implementation ([`MockShuffleCelebornClient`]) allows compilation, unit
//!   testing, and end-to-end shuffle testing without a running Celeborn cluster.
//! * When the real Celeborn Rust/FFI client SDK becomes available, only a single
//!   `impl CelebornClient for ShuffleCelebornClient` block needs to be added; no
//!   upstream code change is required.

use async_trait::async_trait;
use bytes::Bytes;
use common_error::DaftResult;
use futures::stream::BoxStream;

/// Stream of partition data chunks returned from `read_partition`.
///
/// Each chunk is a contiguous slice of bytes in Arrow IPC stream format.
/// Multiple chunks may originate from different map tasks but logically belong
/// to the same reduce partition.
pub type PartitionDataStream = BoxStream<'static, DaftResult<Bytes>>;

/// Connection-level configuration used to construct a [`CelebornClient`].
///
/// Contains all the parameters needed to establish a connection to the
/// Celeborn LifecycleManager. Per-shuffle metadata such as `num_mappers` and
/// `num_partitions` are registered via [`CelebornClient::register_shuffle`]
/// instead, allowing a single client instance to serve multiple shuffles.
#[derive(Clone, Debug)]
pub struct CelebornClientConfig {
    /// LifecycleManager hostname or IP address.
    pub lm_host: String,
    /// LifecycleManager port.
    pub lm_port: i32,
    /// Application-level identifier; usually the Daft query/session id.
    pub app_id: String,
    /// Compression codec for shuffle blocks. One of `"lz4" | "zstd" | "none"`.
    pub compression: String,
}

impl Default for CelebornClientConfig {
    fn default() -> Self {
        Self {
            lm_host: String::new(),
            lm_port: 0,
            app_id: String::new(),
            compression: "lz4".to_string(),
        }
    }
}

/// Abstract Celeborn shuffle client.
///
/// All methods are `async` to accommodate both pure Rust implementations
/// (which may use `tonic` gRPC) and FFI-backed implementations (which may
/// dispatch to a thread pool internally).
///
/// Implementations must be `Send + Sync` so that an `Arc<dyn CelebornClient>`
/// can be shared across the Daft pipeline (multiple Map tasks of the same
/// shuffle share one client instance).
#[async_trait]
pub trait CelebornClient: Send + Sync {
    /// Register a shuffle with the Celeborn cluster.
    ///
    /// Must be called once per shuffle before any `push_data` or `mapper_end`
    /// calls. The client stores `num_mappers` and `num_partitions` internally
    /// so that subsequent per-record calls do not need to repeat them.
    async fn register_shuffle(
        &self,
        shuffle_id: u64,
        num_mappers: u32,
        num_partitions: u32,
    ) -> DaftResult<()>;

    /// Push a single partition payload to the Celeborn cluster.
    ///
    /// `register_shuffle` must have been called for this `shuffle_id` before
    /// calling `push_data`. The client retrieves `num_mappers` and
    /// `num_partitions` from the internally stored metadata.
    ///
    /// * `shuffle_id` - Logical shuffle identifier shared by all mappers/reducers
    ///   participating in this shuffle.
    /// * `map_id` - Index of the current map task, in `[0, num_mappers)`.
    /// * `attempt_id` - Attempt index for the map task; used by Celeborn for
    ///   deduplication when speculative execution is enabled.
    /// * `partition_id` - Target reduce partition index, in `[0, num_partitions)`.
    /// * `data` - Arrow IPC stream bytes for the partition slice.
    async fn push_data(
        &self,
        shuffle_id: u64,
        map_id: u32,
        attempt_id: u32,
        partition_id: u32,
        data: &[u8],
    ) -> DaftResult<()>;

    /// Notify the Celeborn cluster that this map task has finished pushing all
    /// partitions. Must be called exactly once per map attempt.
    async fn mapper_end(&self, shuffle_id: u64, map_id: u32, attempt_id: u32) -> DaftResult<()>;

    /// Read all blocks for a single reduce partition from the Celeborn cluster.
    /// Returns a stream of byte chunks (each chunk is one or more Arrow IPC
    /// record batches).
    async fn read_partition(
        &self,
        shuffle_id: u64,
        partition_id: u32,
    ) -> DaftResult<PartitionDataStream>;

    /// Release all resources (memory, disk, replicas) associated with this
    /// shuffle on the Celeborn cluster. Idempotent; safe to call multiple times.
    async fn unregister_shuffle(&self, shuffle_id: u64) -> DaftResult<()>;
}
