//! Celeborn shuffle client abstraction and implementations.
//!
//! See [`client::CelebornClient`] for the trait definition and
//! [`mock::MockShuffleCelebornClient`] for the in-memory placeholder used during
//! Daft-side development.

use std::sync::Arc;

use common_error::DaftResult;

mod client;
mod ffi;
#[cfg(test)]
mod integration_tests;
mod mock;

pub use client::{CelebornClient, CelebornClientConfig, PartitionDataStream};
pub use ffi::ShuffleCelebornClient;
pub use mock::MockShuffleCelebornClient;

/// Create a connected Celeborn client from connection-level configuration.
///
/// Returns a real FFI-backed [`ShuffleCelebornClient`] that connects to
/// the Celeborn LifecycleManager.
///
/// # Arguments
/// * `config` - Connection-level Celeborn configuration (lm_host, lm_port,
///   app_id, compression).
pub fn connect_celeborn_client(
    config: &CelebornClientConfig,
) -> DaftResult<Arc<dyn CelebornClient>> {
    let client = ShuffleCelebornClient::connect(config)?;
    Ok(Arc::new(client))
}
