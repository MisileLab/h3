//! Memory-mapped storage for zero-copy access
//! 
//! Provides efficient zero-copy access to large vector datasets
//! using memory-mapped files with proper alignment and caching.

use memmap2::{Mmap, MmapMut, MmapOptions};
use std::{
    fs::{File, OpenOptions},
    io::Write,
    path::Path,
};
use crate::{Result, QuantumDBError};

/// Memory-mapped storage for compressed vectors
/// 
/// Provides zero-copy access to compressed vector data with
/// proper alignment for SIMD operations.
pub struct MemoryMappedStorage {
    /// Memory-mapped file for compressed codes
    codes_mmap: Mmap,
    /// Memory-mapped file for metadata
    metadata_mmap: Mmap,
    /// Number of vectors
    num_vectors: usize,
    /// Vector dimension (compressed)
    compressed_dim: usize,
}

impl MemoryMappedStorage {
    /// Create a new memory-mapped storage
    /// 
    /// # Arguments
    /// * `codes_path` - Path to compressed codes file
    /// * `metadata_path` - Path to metadata file
    /// 
    /// # Returns
    /// Memory-mapped storage instance
    pub fn open<P: AsRef<Path>>(codes_path: P, metadata_path: P) -> Result<Self> {
        // Open codes file
        let codes_file = File::open(codes_path)?;
        let codes_mmap = unsafe { Mmap::map(&codes_file)? };
        
        // Open metadata file
        let metadata_file = File::open(metadata_path)?;
        let metadata_mmap = unsafe { Mmap::map(&metadata_file)? };
        
        // Read metadata
        let metadata = IndexMetadata::from_bytes(&metadata_mmap)?;
        
        Ok(Self {
            codes_mmap,
            metadata_mmap,
            num_vectors: metadata.num_vectors,
            compressed_dim: metadata.compressed_dim,
        })
    }
    
    /// Create a new memory-mapped storage with write access
    /// 
    /// # Arguments
    /// * `codes_path` - Path to compressed codes file
    /// * `metadata_path` - Path to metadata file
    /// * `num_vectors` - Number of vectors to store
    /// * `compressed_dim` - Compressed dimension
    /// 
    /// # Returns
    /// Mutable memory-mapped storage instance
    pub fn create<P: AsRef<Path>>(
        codes_path: P,
        metadata_path: P,
        num_vectors: usize,
        compressed_dim: usize,
    ) -> Result<MemoryMappedStorageMut> {
        // Calculate file sizes
        let codes_size = num_vectors * compressed_dim;
        let metadata_size = std::mem::size_of::<IndexMetadata>();
        
        // Create and resize codes file
        let codes_file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .open(&codes_path)?;
        codes_file.set_len(codes_size as u64)?;
        
        // Create metadata file
        let metadata_file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .open(&metadata_path)?;
        metadata_file.set_len(metadata_size as u64)?;
        
        // Memory-map files
        let codes_mmap = unsafe { MmapMut::map_mut(&codes_file)? };
        let metadata_mmap = unsafe { MmapMut::map_mut(&metadata_file)? };
        
        // Initialize metadata
        let metadata = IndexMetadata {
            num_vectors,
            compressed_dim,
            version: 1,
            created_at: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs(),
        };
        
        // Write metadata
        unsafe {
            let metadata_ptr = metadata_mmap.as_mut_ptr() as *mut IndexMetadata;
            std::ptr::write(metadata_ptr, metadata);
        }
        
        Ok(MemoryMappedStorageMut {
            codes_mmap,
            metadata_mmap,
            num_vectors,
            compressed_dim,
        })
    }
    
    /// Get a compressed vector by ID (zero-copy)
    /// 
    /// # Arguments
    /// * `id` - Vector ID
    /// 
    /// # Returns
    /// Slice to compressed vector data
    pub fn get_vector(&self, id: usize) -> Result<&[u8]> {
        if id >= self.num_vectors {
            return Err(QuantumDBError::Index(format!(
                "Vector ID {} out of range (0..{})",
                id, self.num_vectors
            )));
        }
        
        let offset = id * self.compressed_dim;
        let end_offset = offset + self.compressed_dim;
        
        if end_offset > self.codes_mmap.len() {
            return Err(QuantumDBError::Index(format!(
                "Vector offset {} exceeds file size {}",
                end_offset,
                self.codes_mmap.len()
            )));
        }
        
        Ok(&self.codes_mmap[offset..end_offset])
    }
    
    /// Get multiple vectors in parallel
    /// 
    /// # Arguments
    /// * `ids` - Vector IDs
    /// 
    /// # Returns
    /// Vector of slices to compressed vector data
    pub fn get_vectors(&self, ids: &[usize]) -> Result<Vec<&[u8]>> {
        use rayon::prelude::*;
        
        ids.par_iter()
            .map(|&id| self.get_vector(id))
            .collect()
    }
    
    /// Get the number of vectors
    pub fn num_vectors(&self) -> usize {
        self.num_vectors
    }
    
    /// Get the compressed dimension
    pub fn compressed_dim(&self) -> usize {
        self.compressed_dim
    }
    
    /// Get metadata
    pub fn metadata(&self) -> IndexMetadata {
        IndexMetadata::from_bytes(&self.metadata_mmap).unwrap()
    }
}

/// Mutable memory-mapped storage for writing
pub struct MemoryMappedStorageMut {
    codes_mmap: MmapMut,
    metadata_mmap: MmapMut,
    num_vectors: usize,
    compressed_dim: usize,
}

impl MemoryMappedStorageMut {
    /// Set a compressed vector by ID
    /// 
    /// # Arguments
    /// * `id` - Vector ID
    /// * `data` - Compressed vector data
    pub fn set_vector(&mut self, id: usize, data: &[u8]) -> Result<()> {
        if id >= self.num_vectors {
            return Err(QuantumDBError::Index(format!(
                "Vector ID {} out of range (0..{})",
                id, self.num_vectors
            )));
        }
        
        if data.len() != self.compressed_dim {
            return Err(QuantumDBError::Index(format!(
                "Data length {} doesn't match compressed dimension {}",
                data.len(),
                self.compressed_dim
            )));
        }
        
        let offset = id * self.compressed_dim;
        let end_offset = offset + self.compressed_dim;
        
        if end_offset > self.codes_mmap.len() {
            return Err(QuantumDBError::Index(format!(
                "Vector offset {} exceeds file size {}",
                end_offset,
                self.codes_mmap.len()
            )));
        }
        
        unsafe {
            let dst = self.codes_mmap.as_mut_ptr().add(offset);
            std::ptr::copy_nonoverlapping(data.as_ptr(), dst, self.compressed_dim);
        }
        
        Ok(())
    }
    
    /// Flush changes to disk
    pub fn flush(&mut self) -> Result<()> {
        self.codes_mmap.flush()?;
        self.metadata_mmap.flush()?;
        Ok(())
    }
    
    /// Convert to immutable storage
    pub fn into_immutable(self) -> Result<MemoryMappedStorage> {
        // Make sure all changes are flushed
        let mut mutable = self;
        mutable.flush()?;
        
        // Convert to immutable mmaps
        let codes_mmap = unsafe { Mmap::map(&mutable.codes_mmap)?? };
        let metadata_mmap = unsafe { Mmap::map(&mutable.metadata_mmap)?? };
        
        Ok(MemoryMappedStorage {
            codes_mmap,
            metadata_mmap,
            num_vectors: mutable.num_vectors,
            compressed_dim: mutable.compressed_dim,
        })
    }
}

/// Index metadata stored in memory-mapped file
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct IndexMetadata {
    /// Number of vectors in the index
    pub num_vectors: usize,
    /// Compressed dimension
    pub compressed_dim: usize,
    /// Version number
    pub version: u32,
    /// Creation timestamp (Unix epoch)
    pub created_at: u64,
}

impl IndexMetadata {
    /// Read metadata from bytes
    /// 
    /// # Arguments
    /// * `bytes` - Raw metadata bytes
    /// 
    /// # Returns
    /// Parsed metadata
    pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
        if bytes.len() < std::mem::size_of::<IndexMetadata>() {
            return Err(QuantumDBError::Storage(
                std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    "Metadata file too small"
                )
            ));
        }
        
        unsafe {
            let metadata_ptr = bytes.as_ptr() as *const IndexMetadata;
            Ok(std::ptr::read(metadata_ptr))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::tempdir;

    #[test]
    fn test_memory_mapped_storage() {
        let dir = tempdir().unwrap();
        let codes_path = dir.path().join("codes.bin");
        let metadata_path = dir.path().join("metadata.bin");
        
        // Create storage
        let mut storage = MemoryMappedStorage::create(
            &codes_path,
            &metadata_path,
            10,
            16,
        ).unwrap();
        
        // Write some vectors
        for i in 0..10 {
            let data = vec![i as u8; 16];
            storage.set_vector(i, &data).unwrap();
        }
        
        // Flush and convert to immutable
        let storage = storage.into_immutable().unwrap();
        
        // Read vectors back
        for i in 0..10 {
            let vector = storage.get_vector(i).unwrap();
            assert_eq!(vector.len(), 16);
            assert_eq!(vector[0], i as u8);
        }
        
        // Check metadata
        assert_eq!(storage.num_vectors(), 10);
        assert_eq!(storage.compressed_dim(), 16);
    }

    #[test]
    fn test_metadata_serialization() {
        let metadata = IndexMetadata {
            num_vectors: 1000,
            compressed_dim: 16,
            version: 1,
            created_at: 1234567890,
        };
        
        // Serialize to bytes
        let bytes = unsafe {
            std::slice::from_raw_parts(
                &metadata as *const IndexMetadata as *const u8,
                std::mem::size_of::<IndexMetadata>(),
            )
        };
        
        // Deserialize back
        let parsed = IndexMetadata::from_bytes(bytes).unwrap();
        
        assert_eq!(parsed.num_vectors, metadata.num_vectors);
        assert_eq!(parsed.compressed_dim, metadata.compressed_dim);
        assert_eq!(parsed.version, metadata.version);
        assert_eq!(parsed.created_at, metadata.created_at);
    }
}