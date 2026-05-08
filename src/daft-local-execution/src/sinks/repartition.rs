use std::{
    collections::{HashMap, VecDeque},
    sync::{Arc, Mutex},
};

use common_error::DaftResult;
use common_metrics::ops::NodeType;
use daft_core::prelude::SchemaRef;
use daft_dsl::expr::bound_expr::BoundExpr;
use daft_logical_plan::partitioning::RepartitionSpec;
use daft_micropartition::MicroPartition;
use daft_partition_refs::FlightPartitionRef;
#[cfg(feature = "celeborn")]
use daft_shuffles::client::celeborn::CelebornClient;
use daft_shuffles::{
    server::flight_server::ShuffleFlightServer,
    shuffle_cache::{InProgressShuffleCache, partition_ref_id},
};
use itertools::Itertools;
use tracing::{Span, instrument};

use super::{
    blocking_sink::{
        BlockingSink, BlockingSinkFinalizeResult, BlockingSinkOutput, BlockingSinkSinkResult,
    },
    shuffle_metadata::ShufflePartitionMeta,
};
use crate::{
    ExecutionTaskSpawner,
    pipeline::{InputId, NodeName},
};

pub(crate) struct RayRepartitionState {
    states: VecDeque<Vec<MicroPartition>>,
}

impl RayRepartitionState {
    fn push(&mut self, parts: Vec<MicroPartition>) {
        for (vec, part) in self.states.iter_mut().zip(parts) {
            vec.push(part);
        }
    }

    fn emit(&mut self) -> Option<Vec<MicroPartition>> {
        self.states.pop_front()
    }
}

pub(crate) struct FlightRepartitionState {
    partitions: Arc<Vec<Arc<InProgressShuffleCache>>>,
}

impl FlightRepartitionState {
    #[allow(dead_code)]
    async fn push(&self, parts: Vec<MicroPartition>) -> DaftResult<()> {
        let push_futures = self
            .partitions
            .iter()
            .zip(parts)
            .map(|(cache, partition)| cache.push_partition_data(partition));
        futures::future::try_join_all(push_futures).await?;
        Ok(())
    }
}

/// Per-mapper state for the Celeborn backend.
///
/// One instance is created per concurrent input task (i.e. one mapper). It
/// owns: a shared client handle (`client`), the map task identifier
/// (`map_id`, derived from `InputId`), and per-partition counters maintained
/// across `sink()` calls so that `finalize()` can produce
/// `ShufflePartitionMetadata` without a second pass over the data.
#[cfg(feature = "celeborn")]
pub(crate) struct CelebornRepartitionState {
    client: Arc<dyn CelebornClient>,
    shuffle_id: u64,
    map_id: u32,
    attempt_id: u32,
    /// Total number of map tasks for this shuffle (global, from coordinator).
    num_mappers: u32,
    /// Whether `register_shuffle` has been called for this shuffle. The
    /// registration is performed lazily on the first `sink()` call because
    /// `make_state()` is synchronous while `register_shuffle` is async.
    /// Multiple mappers sharing the same `CelebornClient` may race, but
    /// `register_shuffle` is idempotent (it just inserts into a HashMap).
    registered: bool,
    /// Cumulative row count per partition observed by this mapper.
    rows_per_partition: Vec<usize>,
    /// Cumulative byte count per partition observed by this mapper (Arrow IPC
    /// stream bytes actually pushed).
    bytes_per_partition: Vec<usize>,
    num_partitions: u32,
}

pub(crate) enum RepartitionState {
    Ray(RayRepartitionState),
    Flight(FlightRepartitionState),
    #[cfg(feature = "celeborn")]
    Celeborn(CelebornRepartitionState),
}

impl RepartitionState {
    #[allow(dead_code)]
    async fn push(&mut self, parts: Vec<MicroPartition>) -> DaftResult<()> {
        match self {
            Self::Ray(state) => {
                state.push(parts);
                Ok(())
            }
            Self::Flight(state) => state.push(parts).await,
            #[cfg(feature = "celeborn")]
            Self::Celeborn(_) => {
                unreachable!("Celeborn state push is handled in sink() directly")
            }
        }
    }
}

// TODO: unify shuffle backends in all local operations
enum RepartitionBackend {
    Ray,
    Flight {
        shuffle_id: u64,
        shuffle_dirs: Vec<String>,
        compression: Option<String>,
        local_server: Arc<ShuffleFlightServer>,
        shuffle_address: String,
        target_in_memory_size_per_partition: usize,
        schema: SchemaRef,
        // Only accessed from the single-threaded event loop; Mutex is just for Sync.
        partitions: Mutex<HashMap<InputId, Arc<Vec<Arc<InProgressShuffleCache>>>>>,
    },
    #[cfg(feature = "celeborn")]
    Celeborn {
        num_partitions: usize,
        shuffle_id: u64,
        num_mappers: u32,
        repartition_spec: RepartitionSpec,
        client: Arc<dyn CelebornClient>,
    },
}

impl RepartitionBackend {
    fn name(&self) -> &'static str {
        match &self {
            Self::Ray { .. } => "Ray",
            Self::Flight { .. } => "Flight",
            #[cfg(feature = "celeborn")]
            Self::Celeborn { .. } => "Celeborn",
        }
    }
}

pub struct RepartitionSink {
    backend: RepartitionBackend,
    repartition_spec: RepartitionSpec,
    num_partitions: usize,
}

impl RepartitionSink {
    pub fn new_ray(repartition_spec: RepartitionSpec, num_partitions: usize) -> Self {
        Self {
            backend: RepartitionBackend::Ray,
            repartition_spec,
            num_partitions,
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn try_new_flight(
        num_partitions: usize,
        schema: SchemaRef,
        shuffle_id: u64,
        repartition_spec: RepartitionSpec,
        shuffle_dirs: Vec<String>,
        compression: Option<String>,
        local_server: Arc<ShuffleFlightServer>,
        shuffle_address: String,
    ) -> DaftResult<Self> {
        const TARGET_TOTAL_IN_MEMORY_SIZE_BYTES: usize = 1024 * 1024 * 2000;
        Ok(Self {
            backend: RepartitionBackend::Flight {
                shuffle_id,
                shuffle_dirs,
                compression,
                local_server,
                shuffle_address,
                target_in_memory_size_per_partition: (TARGET_TOTAL_IN_MEMORY_SIZE_BYTES
                    / num_partitions)
                    .clamp(1024 * 1024 * 8, 1024 * 1024 * 128),
                schema,
                partitions: Mutex::new(HashMap::new()),
            },
            repartition_spec,
            num_partitions,
        })
    }

    #[allow(clippy::too_many_arguments)]
    #[cfg(feature = "celeborn")]
    pub fn new_celeborn(
        num_partitions: usize,
        shuffle_id: u64,
        num_mappers: u32,
        repartition_spec: RepartitionSpec,
        client: Arc<dyn CelebornClient>,
    ) -> Self {
        Self {
            backend: RepartitionBackend::Celeborn {
                num_partitions,
                shuffle_id,
                num_mappers,
                repartition_spec: repartition_spec.clone(),
                client,
            },
            repartition_spec,
            num_partitions,
        }
    }
}

impl BlockingSink for RepartitionSink {
    type State = RepartitionState;

    #[instrument(skip_all, name = "RepartitionSink::sink")]
    fn sink(
        &self,
        input: MicroPartition,
        state: Self::State,
        _runtime_stats: Arc<Self::Stats>,
        spawner: &ExecutionTaskSpawner,
    ) -> BlockingSinkSinkResult<Self> {
        let repartition_spec = self.repartition_spec.clone();
        let num_partitions = self.num_partitions;

        match (&self.backend, state) {
            (RepartitionBackend::Ray, RepartitionState::Ray(mut state)) => spawner
                .spawn(
                    async move {
                        let partitioned = match repartition_spec {
                            RepartitionSpec::Hash(config) => {
                                let bound_exprs = BoundExpr::bind_all(&config.by, &input.schema())?;
                                input.partition_by_hash(&bound_exprs, num_partitions)?
                            }
                            RepartitionSpec::Random(config) => input
                                .partition_by_random(num_partitions, config.seed.unwrap_or(0))?,
                            RepartitionSpec::Range(config) => input.partition_by_range(
                                &config.by,
                                &config.boundaries,
                                &config.descending,
                            )?,
                        };

                        state.push(partitioned);
                        Ok(RepartitionState::Ray(state))
                    },
                    Span::current(),
                )
                .into(),
            (RepartitionBackend::Flight { .. }, RepartitionState::Flight(state)) => spawner
                .spawn(
                    async move {
                        let partitioned = match repartition_spec {
                            RepartitionSpec::Hash(config) => {
                                let bound_exprs = BoundExpr::bind_all(&config.by, &input.schema())?;
                                input.partition_by_hash(&bound_exprs, num_partitions)?
                            }
                            RepartitionSpec::Random(config) => input
                                .partition_by_random(num_partitions, config.seed.unwrap_or(0))?,
                            RepartitionSpec::Range(config) => input.partition_by_range(
                                &config.by,
                                &config.boundaries,
                                &config.descending,
                            )?,
                        };
                        let push_futures = state
                            .partitions
                            .iter()
                            .zip(partitioned)
                            .map(|(cache, partition)| cache.push_partition_data(partition));
                        futures::future::try_join_all(push_futures).await?;
                        Ok(RepartitionState::Flight(state))
                    },
                    Span::current(),
                )
                .into(),
            #[cfg(feature = "celeborn")]
            (
                RepartitionBackend::Celeborn {
                    repartition_spec,
                    num_partitions,
                    ..
                },
                RepartitionState::Celeborn(mut state),
            ) => {
                let num_partitions = *num_partitions;
                let partition_by = match repartition_spec {
                    RepartitionSpec::Hash(config) => Some(config.by.clone()),
                    RepartitionSpec::Random(_) => None,
                    RepartitionSpec::Range(_) => {
                        unreachable!("Range repartition is not supported for celeborn shuffle")
                    }
                };

                spawner
                    .spawn(
                        async move {
                            // Lazily register the shuffle on the first sink()
                            // call. `make_state` is synchronous so we cannot
                            // call the async `register_shuffle` there.
                            // `register_shuffle` is idempotent, so concurrent
                            // mappers racing here are safe.
                            if !state.registered {
                                state
                                    .client
                                    .register_shuffle(
                                        state.shuffle_id,
                                        state.num_mappers,
                                        state.num_partitions,
                                    )
                                    .await?;
                                state.registered = true;
                            }

                            let partitioned = match &partition_by {
                                Some(partition_by) => {
                                    let partition_by =
                                        BoundExpr::bind_all(partition_by, &input.schema())?;
                                    input.partition_by_hash(&partition_by, num_partitions)?
                                }
                                None => input.partition_by_random(num_partitions, 0)?,
                            };

                            // For each non-empty target partition, serialize as Arrow IPC
                            // stream bytes and push to the Celeborn cluster. Empty
                            // partitions are skipped to avoid wasted RPCs; the reducer
                            // tolerates partitions with zero blocks.
                            for (partition_idx, mp) in partitioned.into_iter().enumerate() {
                                let num_rows = mp.len();
                                if num_rows == 0 {
                                    continue;
                                }
                                let ipc_bytes = mp.write_to_ipc_stream()?;
                                state
                                    .client
                                    .push_data(
                                        state.shuffle_id,
                                        state.map_id,
                                        state.attempt_id,
                                        partition_idx as u32,
                                        &ipc_bytes,
                                    )
                                    .await?;
                                state.rows_per_partition[partition_idx] += num_rows;
                                state.bytes_per_partition[partition_idx] += ipc_bytes.len();
                            }

                            Ok(RepartitionState::Celeborn(state))
                        },
                        Span::current(),
                    )
                    .into()
            }
            _ => panic!("RepartitionSink state/backend mismatch"),
        }
    }

    #[instrument(skip_all, name = "RepartitionSink::finalize")]
    fn finalize(
        &self,
        states: Vec<Self::State>,
        spawner: &ExecutionTaskSpawner,
    ) -> BlockingSinkFinalizeResult {
        match &self.backend {
            RepartitionBackend::Ray => {
                let num_partitions = self.num_partitions;

                let mut states = states
                    .into_iter()
                    .map(|state| match state {
                        RepartitionState::Ray(state) => state,
                        #[cfg(feature = "celeborn")]
                        RepartitionState::Celeborn(_) => {
                            panic!("RepartitionSink state/backend mismatch")
                        }
                        RepartitionState::Flight(_) => {
                            panic!("RepartitionSink state/backend mismatch")
                        }
                    })
                    .collect::<Vec<_>>();

                spawner
                    .spawn(
                        async move {
                            let mut repart_states = states.iter_mut().collect::<Vec<_>>();

                            let mut outputs = Vec::new();
                            for _ in 0..num_partitions {
                                let data = repart_states
                                    .iter_mut()
                                    .flat_map(|state| state.emit().unwrap())
                                    .collect::<Vec<_>>();
                                let fut = tokio::spawn(async move {
                                    let together = MicroPartition::concat(data)?;
                                    let schema = together.schema();
                                    let concated = together.concat_or_get()?;
                                    let mp = MicroPartition::new_loaded(
                                        schema,
                                        Arc::new(if let Some(t) = concated {
                                            vec![t]
                                        } else {
                                            vec![]
                                        }),
                                        None,
                                    );
                                    Ok(mp)
                                });
                                outputs.push(fut);
                            }
                            let partitions = futures::future::try_join_all(outputs)
                                .await
                                .unwrap()
                                .into_iter()
                                .collect::<DaftResult<Vec<_>>>()?;
                            Ok(BlockingSinkOutput::Partitions(partitions))
                        },
                        Span::current(),
                    )
                    .into()
            }
            RepartitionBackend::Flight {
                shuffle_id,
                local_server,
                shuffle_address,
                ..
            } => {
                let shuffle_id = *shuffle_id;
                let local_server = local_server.clone();
                let states = states
                    .into_iter()
                    .map(|state| match state {
                        RepartitionState::Flight(state) => state,
                        #[cfg(feature = "celeborn")]
                        RepartitionState::Celeborn(_) => {
                            panic!("RepartitionSink state/backend mismatch")
                        }
                        RepartitionState::Ray(_) => {
                            panic!("RepartitionSink state/backend mismatch")
                        }
                    })
                    .collect::<Vec<_>>();

                let shuffle_address = shuffle_address.clone();
                spawner
                    .spawn(
                        async move {
                            let partitions = states
                                .into_iter()
                                .next()
                                .expect("Flight repartition finalize requires at least one state")
                                .partitions;
                            let finalized = futures::future::try_join_all(
                                partitions.iter().map(|partition| partition.close()),
                            )
                            .await?;
                            local_server
                                .register_shuffle_partitions(shuffle_id, finalized.clone())
                                .await?;
                            Ok(BlockingSinkOutput::FlightPartitionRefs(
                                finalized
                                    .into_iter()
                                    .map(|partition| FlightPartitionRef {
                                        shuffle_id,
                                        server_address: shuffle_address.clone(),
                                        partition_ref_id: partition.partition_ref_id,
                                        num_rows: partition.num_rows,
                                        size_bytes: partition.size_bytes,
                                    })
                                    .collect(),
                            ))
                        },
                        Span::current(),
                    )
                    .into()
            }
            #[cfg(feature = "celeborn")]
            RepartitionBackend::Celeborn { .. } => {
                let num_partitions = self.num_partitions;
                let states = states
                    .into_iter()
                    .map(|state| match state {
                        #[cfg(feature = "celeborn")]
                        RepartitionState::Celeborn(state) => state,
                        #[cfg(feature = "celeborn")]
                        _ => {
                            panic!("RepartitionSink state/backend mismatch")
                        }
                    })
                    .collect::<Vec<_>>();

                spawner
                    .spawn(
                        async move {
                            // 1. Notify Celeborn that every local mapper attempt has
                            //    finished pushing data. Celeborn requires exactly one
                            //    `mapper_end` per `(shuffle_id, map_id, attempt_id)`.
                            //
                            //    The BlockingSink framework creates `max_concurrency`
                            //    states per input for parallel morsel processing, so
                            //    `states` may contain multiple entries with the same
                            //    `(shuffle_id, map_id, attempt_id)`. We must deduplicate
                            //    to avoid calling mapper_end more than once per mapper,
                            //    which would confuse the Celeborn cluster.
                            let mut seen_mappers = std::collections::HashSet::new();
                            for state in &states {
                                if seen_mappers.insert((
                                    state.shuffle_id,
                                    state.map_id,
                                    state.attempt_id,
                                )) {
                                    state
                                        .client
                                        .mapper_end(
                                            state.shuffle_id,
                                            state.map_id,
                                            state.attempt_id,
                                        )
                                        .await
                                        .map_err(|e| {
                                            tracing::error!(
                                                shuffle_id = state.shuffle_id,
                                                map_id = state.map_id,
                                                attempt_id = state.attempt_id,
                                                error = %e,
                                                "Celeborn mapper_end failed; aborting finalize"
                                            );
                                            e
                                        })?;
                                }
                            }

                            // 2. Aggregate per-partition row/byte counters across all
                            //    local mappers. The resulting metadata represents this
                            //    sink's contribution to the global shuffle output.
                            let mut rows_per_partition = vec![0usize; num_partitions];
                            let mut bytes_per_partition = vec![0usize; num_partitions];
                            for state in states {
                                for (i, count) in state.rows_per_partition.iter().enumerate() {
                                    rows_per_partition[i] += *count;
                                }
                                for (i, count) in state.bytes_per_partition.iter().enumerate() {
                                    bytes_per_partition[i] += *count;
                                }
                            }

                            Ok(BlockingSinkOutput::ShufflePartitionMetas(
                                rows_per_partition
                                    .into_iter()
                                    .zip(bytes_per_partition)
                                    .map(|(num_rows, size_bytes)| {
                                        ShufflePartitionMeta::new(num_rows, size_bytes)
                                    })
                                    .collect(),
                            ))
                        },
                        Span::current(),
                    )
                    .into()
            }
        }
    }

    fn name(&self) -> NodeName {
        format!("Repartition({})", self.backend.name()).into()
    }

    fn op_type(&self) -> NodeType {
        NodeType::Repartition
    }

    fn multiline_display(&self) -> Vec<String> {
        let backend_name = self.backend.name();
        match &self.repartition_spec {
            RepartitionSpec::Hash(config) => vec![format!(
                "Repartition({backend_name}): By {} into {} partitions",
                config.by.iter().map(|e| e.to_string()).join(", "),
                self.num_partitions
            )],
            RepartitionSpec::Random(_) => vec![format!(
                "Repartition({backend_name}): Random into {} partitions",
                self.num_partitions
            )],
            RepartitionSpec::Range(_) => vec![format!(
                "Repartition({backend_name}): Range into {} partitions",
                self.num_partitions
            )],
        }
    }

    fn make_state(&self, input_id: InputId) -> DaftResult<Self::State> {
        match &self.backend {
            RepartitionBackend::Ray => Ok(RepartitionState::Ray(RayRepartitionState {
                states: (0..self.num_partitions).map(|_| vec![]).collect(),
            })),
            RepartitionBackend::Flight {
                shuffle_dirs,
                shuffle_id,
                target_in_memory_size_per_partition,
                compression,
                schema,
                partitions,
                ..
            } => {
                let mut partitions = partitions.lock().unwrap();
                let partition_set = match partitions.entry(input_id) {
                    std::collections::hash_map::Entry::Occupied(e) => e.get().clone(),
                    std::collections::hash_map::Entry::Vacant(e) => {
                        let partition_set = Arc::new(
                            (0..self.num_partitions)
                                .map(|partition_idx| {
                                    Ok(Arc::new(InProgressShuffleCache::try_new(
                                        partition_ref_id(input_id, partition_idx),
                                        schema.clone(),
                                        shuffle_dirs,
                                        *shuffle_id,
                                        *target_in_memory_size_per_partition,
                                        compression.as_deref(),
                                    )?))
                                })
                                .collect::<DaftResult<Vec<_>>>()?,
                        );
                        e.insert(partition_set.clone());
                        partition_set
                    }
                };
                Ok(RepartitionState::Flight(FlightRepartitionState {
                    partitions: partition_set,
                }))
            }
            #[cfg(feature = "celeborn")]
            RepartitionBackend::Celeborn {
                num_partitions,
                shuffle_id,
                num_mappers,
                client,
                ..
            } => {
                // `InputId` is u32 in the local-execution layer; map it directly
                // to Celeborn's `map_id`. `attempt_id` is fixed at 0 because Daft
                // does not currently support speculative execution — retry attempts
                // are handled separately by the scheduler, so this is always 0.
                Ok(RepartitionState::Celeborn(CelebornRepartitionState {
                    client: client.clone(),
                    shuffle_id: *shuffle_id,
                    map_id: input_id,
                    attempt_id: 0,
                    num_mappers: *num_mappers,
                    registered: false,
                    rows_per_partition: vec![0; *num_partitions],
                    bytes_per_partition: vec![0; *num_partitions],
                    num_partitions: *num_partitions as u32,
                }))
            }
        }
    }
}
