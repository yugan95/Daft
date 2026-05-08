//! FFI-backed Celeborn shuffle client implementation.
//!
//! This bridges our async [`CelebornClient`] trait to the synchronous
//! `celeborn_client::ShuffleClient` (C++ FFI). All FFI calls are dispatched
//! to a blocking thread via `tokio::task::spawn_blocking` so they never block
//! the async runtime.

use std::{
    collections::HashMap,
    io::{self, BufReader, Read},
    sync::{Arc, Mutex},
};

use async_trait::async_trait;
use bytes::Bytes;
use celeborn_client::{Config as CelebornConfig, ShuffleClient};
use common_error::{DaftError, DaftResult};

use super::client::{CelebornClient, CelebornClientConfig, PartitionDataStream};

/// A reader wrapper that records all bytes passing through `read()` into an
/// internal buffer, allowing Arrow's `StreamReader` to drive IPC parsing
/// while we capture the raw bytes of each self-contained IPC stream.
struct TeeReader<R> {
    inner: R,
    buffer: Vec<u8>,
}

impl<R: Read> Read for TeeReader<R> {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        let n = self.inner.read(buf)?;
        self.buffer.extend_from_slice(&buf[..n]);
        Ok(n)
    }
}

/// Convert a value to `i32`, returning a descriptive error on overflow.
///
/// The Celeborn C++ FFI uses `i32` for all ID parameters while the Daft
/// trait uses wider unsigned types. This helper centralises the checked
/// conversion so callers don't repeat the same boilerplate.
fn to_ffi_i32(value: impl TryInto<i32> + std::fmt::Display + Copy, name: &str) -> DaftResult<i32> {
    value.try_into().map_err(|_| {
        DaftError::External(format!("{name} {value} overflows i32 (Celeborn FFI limit)").into())
    })
}

/// Run a synchronous FFI closure on the tokio blocking thread pool and
/// map JoinError (panic) into a [`DaftError`].
async fn run_blocking<F, R>(op_name: &str, f: F) -> DaftResult<R>
where
    F: FnOnce() -> DaftResult<R> + Send + 'static,
    R: Send + 'static,
{
    let op = op_name.to_owned();
    tokio::task::spawn_blocking(f)
        .await
        .map_err(|e| DaftError::External(format!("Celeborn {op} task panicked: {e}").into()))?
}

/// Thread-safe wrapper around `celeborn_client::ShuffleClient`.
///
/// `ShuffleClient` holds an opaque raw pointer to the C++ FFI handle, which
/// prevents auto `Send`/`Sync`. As of the celeborn-client "parallel read/write"
/// revision, every `ShuffleClient` method takes `&self` and the underlying C++
/// `ShuffleClientImpl` synchronises all shared state internally (folly
/// concurrent maps, per-shuffle registration mutex, per-call compressor).
/// Concurrent `push_data` / `read_partition` from multiple threads is therefore
/// safe, so this wrapper is both `Send` and `Sync` and can be shared via a
/// plain `Arc` without any external lock.
struct CelebornShuffleClient(ShuffleClient);

// SAFETY: `ShuffleClient` is `!Send`/`!Sync` at the type level only because it
// holds an opaque raw pointer to the C++ FFI handle. That handle owns its own
// thread pool and synchronises every shared structure internally, and all
// methods take `&self`, so it is safe to both move the wrapper between threads
// (`Send`) and share `&CelebornShuffleClient` across threads (`Sync`). This
// mirrors the `unsafe impl Send + Sync for ShuffleClient` in celeborn-client
// itself.
#[allow(clippy::non_send_fields_in_send_ty)]
unsafe impl Send for CelebornShuffleClient {}
// SAFETY: see the `Send` impl above; the inner C++ client synchronises all
// shared state internally and exposes every operation through `&self`.
unsafe impl Sync for CelebornShuffleClient {}

/// Celeborn shuffle client backed by the C++ FFI implementation.
///
/// Thread-safety: as of the celeborn-client "parallel read/write" revision the
/// underlying `ShuffleClient` exposes every operation through `&self` and
/// synchronises internally, so we share it via a plain `Arc` with **no external
/// lock**. This lets multiple partitions be pushed and read truly concurrently.
///
/// This is a **connection-level** object: one instance per Worker, shared
/// across all shuffles. Per-shuffle metadata (`num_mappers`, `num_partitions`)
/// is stored in `shuffle_meta`, which is still guarded by a `Mutex` because it
/// is plain Rust state mutated by `register_shuffle` / `unregister_shuffle`.
pub struct ShuffleCelebornClient {
    inner: Arc<CelebornShuffleClient>,
    shuffle_meta: Arc<Mutex<HashMap<u64, (u32, u32)>>>,
}

// SAFETY: both fields are `Arc<T>` where `T: Send + Sync`
// (`CelebornShuffleClient` via its manual impls above, `Mutex<HashMap<..>>`
// inherently), so the struct is safe to send and share across threads.
unsafe impl Send for ShuffleCelebornClient {}
unsafe impl Sync for ShuffleCelebornClient {}

impl ShuffleCelebornClient {
    /// Connect to a running Celeborn LifecycleManager and return a new client.
    ///
    /// # Arguments
    /// * `config` - Connection-level Celeborn configuration (lm_host, lm_port,
    ///   app_id, compression).
    pub fn connect(config: &CelebornClientConfig) -> DaftResult<Self> {
        let codec = config.compression.to_uppercase();
        let celeborn_config = CelebornConfig {
            app_id: config.app_id.clone(),
            push_buffer_max_size: 0, // use C++ default (64KB)
            shuffle_compression_codec: codec,
        };

        let client = ShuffleClient::connect(celeborn_config, &config.lm_host, config.lm_port)
            .map_err(|e| {
                DaftError::External(
                    format!(
                        "Failed to connect to Celeborn LifecycleManager at {}:{}: {e}",
                        config.lm_host, config.lm_port
                    )
                    .into(),
                )
            })?;

        Ok(Self {
            inner: Arc::new(CelebornShuffleClient(client)),
            shuffle_meta: Arc::new(Mutex::new(HashMap::new())),
        })
    }
}

#[async_trait]
impl CelebornClient for ShuffleCelebornClient {
    async fn register_shuffle(
        &self,
        shuffle_id: u64,
        num_mappers: u32,
        num_partitions: u32,
    ) -> DaftResult<()> {
        let mut meta = self
            .shuffle_meta
            .lock()
            .map_err(|e| DaftError::External(format!("shuffle_meta lock poisoned: {e}").into()))?;
        meta.insert(shuffle_id, (num_mappers, num_partitions));
        Ok(())
    }

    async fn push_data(
        &self,
        shuffle_id: u64,
        map_id: u32,
        attempt_id: u32,
        partition_id: u32,
        data: &[u8],
    ) -> DaftResult<()> {
        let (num_mappers_raw, num_partitions_raw) = {
            let meta = self.shuffle_meta.lock().map_err(|e| {
                DaftError::External(format!("shuffle_meta lock poisoned: {e}").into())
            })?;
            *meta.get(&shuffle_id).ok_or_else(|| {
                DaftError::External(
                    format!("shuffle {shuffle_id} not registered; call register_shuffle first")
                        .into(),
                )
            })?
        };
        let inner = Arc::clone(&self.inner);
        let shuffle_id = to_ffi_i32(shuffle_id, "shuffle_id")?;
        let map_id = to_ffi_i32(map_id, "map_id")?;
        let attempt_id = to_ffi_i32(attempt_id, "attempt_id")?;
        let partition_id = to_ffi_i32(partition_id, "partition_id")?;
        let num_mappers = to_ffi_i32(num_mappers_raw, "num_mappers")?;
        let num_partitions = to_ffi_i32(num_partitions_raw, "num_partitions")?;
        let data_owned = data.to_vec();

        run_blocking("push_data", move || {
            inner
                .0
                .push_data(
                    shuffle_id,
                    map_id,
                    attempt_id,
                    partition_id,
                    &data_owned,
                    num_mappers,
                    num_partitions,
                )
                .map_err(|e| DaftError::External(format!("Celeborn push_data failed: {e}").into()))
        })
        .await
    }

    async fn mapper_end(&self, shuffle_id: u64, map_id: u32, attempt_id: u32) -> DaftResult<()> {
        let (num_mappers_raw, _) = {
            let meta = self.shuffle_meta.lock().map_err(|e| {
                DaftError::External(format!("shuffle_meta lock poisoned: {e}").into())
            })?;
            *meta.get(&shuffle_id).ok_or_else(|| {
                DaftError::External(
                    format!("shuffle {shuffle_id} not registered; call register_shuffle first")
                        .into(),
                )
            })?
        };
        let inner = Arc::clone(&self.inner);
        let shuffle_id = to_ffi_i32(shuffle_id, "shuffle_id")?;
        let map_id = to_ffi_i32(map_id, "map_id")?;
        let attempt_id = to_ffi_i32(attempt_id, "attempt_id")?;
        let num_mappers = to_ffi_i32(num_mappers_raw, "num_mappers")?;

        run_blocking("mapper_end", move || {
            inner
                .0
                .mapper_end(shuffle_id, map_id, attempt_id, num_mappers)
                .map_err(|e| DaftError::External(format!("Celeborn mapper_end failed: {e}").into()))
        })
        .await
    }

    async fn read_partition(
        &self,
        shuffle_id: u64,
        partition_id: u32,
    ) -> DaftResult<PartitionDataStream> {
        let (num_mappers_raw, _) = {
            let meta = self.shuffle_meta.lock().map_err(|e| {
                DaftError::External(format!("shuffle_meta lock poisoned: {e}").into())
            })?;
            *meta.get(&shuffle_id).ok_or_else(|| {
                DaftError::External(
                    format!("shuffle {shuffle_id} not registered; call register_shuffle first")
                        .into(),
                )
            })?
        };
        let inner = Arc::clone(&self.inner);
        let shuffle_id_ffi = to_ffi_i32(shuffle_id, "shuffle_id")?;
        let partition_id = to_ffi_i32(partition_id, "partition_id")?;
        let num_mappers = to_ffi_i32(num_mappers_raw, "num_mappers")?;

        let (tx, rx) = async_channel::bounded(4);

        tokio::task::spawn_blocking(move || {
            let run = || -> DaftResult<()> {
                // No external lock: the celeborn-client `ShuffleClient` takes
                // `&self` and synchronises internally, so multiple partitions
                // can be opened and read concurrently from different blocking
                // threads sharing the same `Arc<CelebornShuffleClient>`.
                inner
                    .0
                    .update_reducer_file_group(shuffle_id_ffi)
                    .map_err(|e| {
                        DaftError::External(
                            format!("Celeborn update_reducer_file_group failed: {e}").into(),
                        )
                    })?;

                let reader = inner
                    .0
                    .open_partition(shuffle_id_ffi, partition_id, 0, 0, num_mappers)
                    .map_err(|e| {
                        DaftError::External(format!("Celeborn open_partition failed: {e}").into())
                    })?;

                let mut tee = TeeReader {
                    inner: BufReader::with_capacity(64 * 1024, reader),
                    buffer: Vec::new(),
                };

                loop {
                    tee.buffer.clear();
                    match arrow_ipc::reader::StreamReader::try_new(&mut tee, None) {
                        Ok(stream_reader) => {
                            for batch in stream_reader {
                                batch.map_err(|e| {
                                    DaftError::External(
                                        format!("Celeborn IPC stream decode error: {e}").into(),
                                    )
                                })?;
                            }
                            let ipc_bytes = Bytes::from(std::mem::take(&mut tee.buffer));
                            if !ipc_bytes.is_empty() && tx.send_blocking(Ok(ipc_bytes)).is_err() {
                                break;
                            }
                        }
                        Err(e) => {
                            if !tee.buffer.is_empty() {
                                let preview_len = tee.buffer.len().min(64);
                                let _ = tx.send_blocking(Err(DaftError::External(
                                    format!(
                                        "Celeborn: corrupt Arrow IPC stream header in partition data \
                                         (shuffle_id={shuffle_id_ffi}, partition_id={partition_id}, \
                                         num_mappers={num_mappers}, buffer_len={}, first_{preview_len}_bytes={:?}, \
                                         ipc_error={e})",
                                        tee.buffer.len(),
                                        &tee.buffer[..preview_len],
                                    )
                                    .into(),
                                )));
                            }
                            break;
                        }
                    }
                }
                Ok(())
            };

            if let Err(e) = run() {
                let _ = tx.send_blocking(Err(e));
            }
        });

        Ok(Box::pin(rx))
    }

    /// Clean up local per-shuffle metadata.
    ///
    /// The underlying `celeborn_client::ShuffleClient` C++ FFI does not expose
    /// an explicit `unregister_shuffle` API, so server-side cleanup relies on
    /// the Celeborn cluster's own garbage-collection mechanism
    /// (LifecycleManager timeout / application heartbeat expiry). However, we
    /// still remove the local `shuffle_meta` entry to avoid unbounded memory
    /// growth when many shuffles are executed through the same client instance.
    async fn unregister_shuffle(&self, shuffle_id: u64) -> DaftResult<()> {
        let mut meta = self
            .shuffle_meta
            .lock()
            .map_err(|e| DaftError::External(format!("shuffle_meta lock poisoned: {e}").into()))?;
        meta.remove(&shuffle_id);
        Ok(())
    }
}
