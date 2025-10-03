//! Storage layer for QuantumDB
//! 
//! Implements memory-mapped storage with SafeTensors for zero-copy
//! loading and efficient persistence of compressed vectors and indices.

pub mod mmap;
pub mod persistence;
pub mod quantumdb;

pub use mmap::MemoryMappedStorage;
pub use persistence::{SafeTensorsStorage, IndexMetadata};
pub use quantumdb::QuantumDB;