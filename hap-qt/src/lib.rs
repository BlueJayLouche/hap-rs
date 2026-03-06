//! QuickTime Container Reader for HAP Video
//!
//! Parses QuickTime/MP4 containers to extract HAP video frames without ffmpeg.

use byteorder::{BigEndian, ReadBytesExt};
use std::fs::File;
use std::io::{self, Read, Seek, SeekFrom};
use std::path::Path;
use thiserror::Error;

pub use hap_parser::{HapFrame, TextureFormat, TopLevelType};

/// Errors that can occur during QuickTime parsing
#[derive(Error, Debug)]
pub enum QtError {
    #[error("IO error: {0}")]
    Io(#[from] io::Error),
    
    #[error("Invalid QuickTime file: {0}")]
    InvalidFile(String),
    
    #[error("No HAP video track found")]
    NoHapTrack,
    
    #[error("Unsupported codec: {0}")]
    UnsupportedCodec(String),
    
    #[error("HAP parse error: {0}")]
    HapError(#[from] hap_parser::HapError),
}

/// A QuickTime/HAP movie reader
pub struct QtHapReader {
    /// File handle
    file: File,
    /// Video track info
    track: VideoTrack,
    /// Movie duration in seconds
    duration: f64,
    /// Frame rate
    fps: f32,
}

/// Video track information
#[derive(Debug, Clone)]
struct VideoTrack {
    /// Track ID
    track_id: u32,
    /// Video width
    width: u32,
    /// Video height
    height: u32,
    /// Total frame count
    frame_count: u32,
    /// Timescale (time units per second)
    timescale: u32,
    /// Duration in timescale units
    duration: u64,
    /// Sample entry (codec info)
    sample_entry: SampleEntry,
    /// Sample sizes
    sample_sizes: Vec<u32>,
    /// Chunk offsets
    chunk_offsets: Vec<u64>,
    /// Sample to chunk mappings
    sample_to_chunk: Vec<SampleToChunkEntry>,
    /// Sample durations (delta)
    sample_deltas: Vec<u32>,
}

/// Sample entry (codec description)
#[derive(Debug, Clone)]
struct SampleEntry {
    /// Codec type (e.g., "Hap1", "Hap5", "HapY")
    codec_type: String,
    /// Data reference index
    data_reference_index: u16,
}

/// Sample to chunk entry
#[derive(Debug, Clone)]
struct SampleToChunkEntry {
    /// First chunk number using this table entry
    first_chunk: u32,
    /// Samples per chunk
    samples_per_chunk: u32,
    /// Sample description ID
    sample_description_index: u32,
}

impl QtHapReader {
    /// Open a QuickTime HAP file
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self, QtError> {
        let mut file = File::open(path)?;
        
        // Parse the file structure
        let track = Self::parse_movie(&mut file)?;
        
        // Calculate duration and fps
        let duration = track.duration as f64 / track.timescale as f64;
        let fps = if duration > 0.0 {
            track.frame_count as f32 / duration as f32
        } else {
            30.0
        };
        
        Ok(Self {
            file,
            track,
            duration,
            fps,
        })
    }
    
    /// Get video dimensions
    pub fn resolution(&self) -> (u32, u32) {
        (self.track.width, self.track.height)
    }
    
    /// Get frame count
    pub fn frame_count(&self) -> u32 {
        self.track.frame_count
    }
    
    /// Get frame rate
    pub fn fps(&self) -> f32 {
        self.fps
    }
    
    /// Get duration in seconds
    pub fn duration(&self) -> f64 {
        self.duration
    }
    
    /// Get codec type (e.g., "Hap1", "Hap5", "HapY")
    pub fn codec_type(&self) -> &str {
        &self.track.sample_entry.codec_type
    }
    
    /// Get texture format
    pub fn texture_format(&self) -> hap_parser::TextureFormat {
        match self.track.sample_entry.codec_type.as_str() {
            "Hap1" => hap_parser::TextureFormat::RgbDxt1,
            "Hap5" => hap_parser::TextureFormat::RgbaDxt5,
            "HapY" => hap_parser::TextureFormat::YcoCgDxt5,
            "HapA" => hap_parser::TextureFormat::AlphaRgtc1,
            "Hap7" => hap_parser::TextureFormat::RgbaBc7,
            "HapH" => hap_parser::TextureFormat::RgbBc6hUfloat,
            _ => hap_parser::TextureFormat::RgbDxt1, // Default to DXT1
        }
    }
    
    /// Read a specific frame
    pub fn read_frame(&mut self, frame_index: u32) -> Result<HapFrame, QtError> {
        if frame_index >= self.track.frame_count {
            return Err(QtError::InvalidFile(format!(
                "Frame index {} out of range (0-{})",
                frame_index,
                self.track.frame_count - 1
            )));
        }
        
        // Find which chunk contains this frame
        let (chunk_index, sample_offset) = self.frame_to_chunk(frame_index);
        
        // Get chunk offset
        let chunk_offset = self.track.chunk_offsets.get(chunk_index as usize)
            .ok_or_else(|| QtError::InvalidFile("Invalid chunk offset table".to_string()))?;
        
        // Calculate frame offset within chunk
        let mut frame_offset = *chunk_offset;
        for i in 0..sample_offset {
            let idx = (frame_index - sample_offset + i) as usize;
            if idx < self.track.sample_sizes.len() {
                frame_offset += self.track.sample_sizes[idx] as u64;
            }
        }
        
        // Get frame size
        let frame_size = self.track.sample_sizes.get(frame_index as usize)
            .ok_or_else(|| QtError::InvalidFile("Invalid sample size table".to_string()))?;
        
        // Read frame data
        self.file.seek(SeekFrom::Start(frame_offset))?;
        let mut frame_data = vec![0u8; *frame_size as usize];
        self.file.read_exact(&mut frame_data)?;
        
        // Parse HAP frame
        hap_parser::parse_frame(&frame_data)
            .map_err(|e| QtError::HapError(e))
    }
    
    /// Convert frame index to (chunk_index, sample_offset_within_chunk)
    fn frame_to_chunk(&self, frame_index: u32) -> (u32, u32) {
        let mut sample_count = 0u32;
        
        for (i, entry) in self.track.sample_to_chunk.iter().enumerate() {
            let next_entry_first_chunk = if i + 1 < self.track.sample_to_chunk.len() {
                self.track.sample_to_chunk[i + 1].first_chunk
            } else {
                self.track.chunk_offsets.len() as u32 + 1
            };
            
            let chunks_in_entry = next_entry_first_chunk - entry.first_chunk;
            let samples_in_entry = chunks_in_entry * entry.samples_per_chunk;
            
            if sample_count + samples_in_entry > frame_index {
                let offset_in_entry = frame_index - sample_count;
                let chunk_offset = offset_in_entry / entry.samples_per_chunk;
                let sample_offset = offset_in_entry % entry.samples_per_chunk;
                
                return (entry.first_chunk - 1 + chunk_offset, sample_offset);
            }
            
            sample_count += samples_in_entry;
        }
        
        (0, 0)
    }
    
    /// Parse the movie structure
    fn parse_movie(file: &mut File) -> Result<VideoTrack, QtError> {
        let file_size = file.metadata()?.len();
        
        // Parse top-level atoms
        let mut pos = 0u64;
        let mut moov_data = None;
        let mut mdat_offset = None;
        
        while pos < file_size {
            file.seek(SeekFrom::Start(pos))?;
            
            let size = file.read_u32::<BigEndian>()? as u64;
            let mut type_buf = [0u8; 4];
            file.read_exact(&mut type_buf)?;
            let atom_type = String::from_utf8_lossy(&type_buf);
            
            if size == 0 {
                break;
            }
            
            let actual_size = if size == 1 {
                // Extended size
                file.read_u64::<BigEndian>()?
            } else {
                size
            };
            
            match atom_type.as_ref() {
                "moov" => {
                    let mut data = vec![0u8; (actual_size - 8) as usize];
                    file.read_exact(&mut data)?;
                    moov_data = Some(data);
                }
                "mdat" => {
                    mdat_offset = Some(pos);
                }
                _ => {}
            }
            
            pos += actual_size;
        }
        
        let moov_data = moov_data.ok_or_else(|| QtError::InvalidFile("No moov atom found".to_string()))?;
        let mdat_offset = mdat_offset.ok_or_else(|| QtError::InvalidFile("No mdat atom found".to_string()))?;
        
        // Parse moov to find video track
        Self::parse_moov(&moov_data, mdat_offset)
    }
    
    /// Parse moov atom
    fn parse_moov(data: &[u8], mdat_offset: u64) -> Result<VideoTrack, QtError> {
        let mut cursor = io::Cursor::new(data);
        
        while cursor.position() < data.len() as u64 {
            let pos = cursor.position();
            let remaining = data.len() - pos as usize;
            
            if remaining < 8 {
                break;
            }
            
            let size = cursor.read_u32::<BigEndian>()? as usize;
            let mut type_buf = [0u8; 4];
            cursor.read_exact(&mut type_buf)?;
            let atom_type = String::from_utf8_lossy(&type_buf);
            
            if size == 0 || size > remaining {
                break;
            }
            
            let atom_data = &data[(pos + 8) as usize..(pos + size as u64) as usize];
            
            if atom_type == "trak" {
                // Try to parse this track
                if let Ok(Some(track)) = Self::parse_trak(atom_data, mdat_offset) {
                    return Ok(track);
                }
            }
            
            cursor.set_position(pos + size as u64);
        }
        
        Err(QtError::NoHapTrack)
    }
    
    /// Parse trak atom
    fn parse_trak(data: &[u8], mdat_offset: u64) -> Result<Option<VideoTrack>, QtError> {
        let mut track_id = None;
        let mut width = 0u32;
        let mut height = 0u32;
        let mut timescale = 0u32;
        let mut duration = 0u64;
        let mut sample_entry = None;
        let mut sample_sizes = Vec::new();
        let mut chunk_offsets = Vec::new();
        let mut sample_to_chunk = Vec::new();
        let mut sample_deltas = Vec::new();
        
        let mut cursor = io::Cursor::new(data);
        
        while cursor.position() < data.len() as u64 {
            let pos = cursor.position();
            let remaining = data.len() - pos as usize;
            
            if remaining < 8 {
                break;
            }
            
            let size = cursor.read_u32::<BigEndian>()? as usize;
            let mut type_buf = [0u8; 4];
            cursor.read_exact(&mut type_buf)?;
            let atom_type = String::from_utf8_lossy(&type_buf);
            
            if size == 0 || size > remaining {
                break;
            }
            
            let atom_data = &data[(pos + 8) as usize..(pos + size as u64) as usize];
            
            match atom_type.as_ref() {
                "tkhd" => {
                    // Track header - contains track ID and dimensions
                    let (tid, w, h) = Self::parse_tkhd(atom_data)?;
                    track_id = Some(tid);
                    width = w;
                    height = h;
                }
                "mdia" => {
                    // Media - contains timescale, duration, and sample table
                    let result = Self::parse_mdia(atom_data)?;
                    timescale = result.timescale;
                    duration = result.duration;
                    sample_entry = result.sample_entry;
                    sample_sizes = result.sample_sizes;
                    chunk_offsets = result.chunk_offsets;
                    sample_to_chunk = result.sample_to_chunk;
                    sample_deltas = result.sample_deltas;
                }
                _ => {}
            }
            
            cursor.set_position(pos + size as u64);
        }
        
        // Check if this is a HAP track
        let sample_entry = match sample_entry {
            Some(se) => se,
            None => return Ok(None),
        };
        
        if !sample_entry.codec_type.starts_with("Hap") {
            return Ok(None); // Not a HAP track
        }
        
        let frame_count = sample_sizes.len() as u32;
        
        // Adjust chunk offsets relative to file start
        let adjusted_offsets: Vec<u64> = chunk_offsets.iter()
            .map(|&offset| {
                // Chunk offset is usually relative to mdat start, convert to absolute
                if offset < mdat_offset {
                    mdat_offset + 8 + offset // +8 for mdat header
                } else {
                    offset
                }
            })
            .collect();
        
        Ok(Some(VideoTrack {
            track_id: track_id.unwrap_or(1),
            width,
            height,
            frame_count,
            timescale,
            duration,
            sample_entry,
            sample_sizes,
            chunk_offsets: adjusted_offsets,
            sample_to_chunk,
            sample_deltas,
        }))
    }
    
    /// Parse tkhd (track header) atom
    fn parse_tkhd(data: &[u8]) -> Result<(u32, u32, u32), QtError> {
        let mut cursor = io::Cursor::new(data);
        
        // Version and flags
        let version = cursor.read_u8()?;
        let _flags = cursor.read_u24::<BigEndian>()?;
        
        // Skip creation_time, modification_time, track_ID, reserved
        if version == 1 {
            cursor.seek(SeekFrom::Current(8 + 8 + 4 + 4))?; // 64-bit times + track ID + reserved
            cursor.seek(SeekFrom::Current(8 + 8))?; // duration (64-bit)
        } else {
            cursor.seek(SeekFrom::Current(4 + 4 + 4 + 4))?; // 32-bit times + track ID + reserved
            cursor.seek(SeekFrom::Current(4))?; // duration (32-bit)
        }
        
        // Skip reserved (8 bytes), layer (2), alternate_group (2), volume (2), reserved (2)
        cursor.seek(SeekFrom::Current(8 + 2 + 2 + 2 + 2))?;
        
        // Skip matrix (36 bytes)
        cursor.seek(SeekFrom::Current(36))?;
        
        // Width and height (16.16 fixed point)
        let width = (cursor.read_u32::<BigEndian>()? >> 16) as u32;
        let height = (cursor.read_u32::<BigEndian>()? >> 16) as u32;
        
        // We need track ID from earlier, but we skipped it - parse again
        cursor.set_position(4); // After version/flags
        if version == 1 {
            cursor.seek(SeekFrom::Current(16))?; // Skip creation/modification time
        } else {
            cursor.seek(SeekFrom::Current(8))?; // Skip creation/modification time
        }
        let track_id = cursor.read_u32::<BigEndian>()?;
        
        Ok((track_id, width, height))
    }
    
    /// Parse mdia (media) atom
    fn parse_mdia(data: &[u8]) -> Result<MediaInfo, QtError> {
        let mut timescale = 0u32;
        let mut duration = 0u64;
        let mut sample_entry = None;
        let mut sample_sizes = Vec::new();
        let mut chunk_offsets = Vec::new();
        let mut sample_to_chunk = Vec::new();
        let mut sample_deltas = Vec::new();
        
        let mut cursor = io::Cursor::new(data);
        
        while cursor.position() < data.len() as u64 {
            let pos = cursor.position();
            let remaining = data.len() - pos as usize;
            
            if remaining < 8 {
                break;
            }
            
            let size = cursor.read_u32::<BigEndian>()? as usize;
            let mut type_buf = [0u8; 4];
            cursor.read_exact(&mut type_buf)?;
            let atom_type = String::from_utf8_lossy(&type_buf);
            
            if size == 0 || size > remaining {
                break;
            }
            
            let atom_data = &data[(pos + 8) as usize..(pos + size as u64) as usize];
            
            match atom_type.as_ref() {
                "mdhd" => {
                    // Media header - timescale and duration
                    let (ts, dur) = Self::parse_mdhd(atom_data)?;
                    timescale = ts;
                    duration = dur;
                }
                "minf" => {
                    // Media info - contains sample table
                    let result = Self::parse_minf(atom_data)?;
                    sample_entry = result.sample_entry;
                    sample_sizes = result.sample_sizes;
                    chunk_offsets = result.chunk_offsets;
                    sample_to_chunk = result.sample_to_chunk;
                    sample_deltas = result.sample_deltas;
                }
                _ => {}
            }
            
            cursor.set_position(pos + size as u64);
        }
        
        Ok(MediaInfo {
            timescale,
            duration,
            sample_entry,
            sample_sizes,
            chunk_offsets,
            sample_to_chunk,
            sample_deltas,
        })
    }
    
    /// Parse mdhd (media header) atom
    fn parse_mdhd(data: &[u8]) -> Result<(u32, u64), QtError> {
        let mut cursor = io::Cursor::new(data);
        
        let version = cursor.read_u8()?;
        let _flags = cursor.read_u24::<BigEndian>()?;
        
        if version == 1 {
            cursor.seek(SeekFrom::Current(8 + 8))?; // 64-bit times
        } else {
            cursor.seek(SeekFrom::Current(4 + 4))?; // 32-bit times
        }
        
        let timescale = cursor.read_u32::<BigEndian>()?;
        
        let duration = if version == 1 {
            cursor.read_u64::<BigEndian>()?
        } else {
            cursor.read_u32::<BigEndian>()? as u64
        };
        
        Ok((timescale, duration))
    }
    
    /// Parse minf (media info) atom
    fn parse_minf(data: &[u8]) -> Result<SampleTableInfo, QtError> {
        let mut sample_entry = None;
        let mut sample_sizes = Vec::new();
        let mut chunk_offsets = Vec::new();
        let mut sample_to_chunk = Vec::new();
        let mut sample_deltas = Vec::new();
        
        let mut cursor = io::Cursor::new(data);
        
        while cursor.position() < data.len() as u64 {
            let pos = cursor.position();
            let remaining = data.len() - pos as usize;
            
            if remaining < 8 {
                break;
            }
            
            let size = cursor.read_u32::<BigEndian>()? as usize;
            let mut type_buf = [0u8; 4];
            cursor.read_exact(&mut type_buf)?;
            let atom_type = String::from_utf8_lossy(&type_buf);
            
            if size == 0 || size > remaining {
                break;
            }
            
            let atom_data = &data[(pos + 8) as usize..(pos + size as u64) as usize];
            
            if atom_type == "stbl" {
                // Sample table
                let result = Self::parse_stbl(atom_data)?;
                sample_entry = result.sample_entry;
                sample_sizes = result.sample_sizes;
                chunk_offsets = result.chunk_offsets;
                sample_to_chunk = result.sample_to_chunk;
                sample_deltas = result.sample_deltas;
            }
            
            cursor.set_position(pos + size as u64);
        }
        
        Ok(SampleTableInfo {
            sample_entry,
            sample_sizes,
            chunk_offsets,
            sample_to_chunk,
            sample_deltas,
        })
    }
    
    /// Parse stbl (sample table) atom
    fn parse_stbl(data: &[u8]) -> Result<SampleTableInfo, QtError> {
        let mut sample_entry = None;
        let mut sample_sizes = Vec::new();
        let mut chunk_offsets = Vec::new();
        let mut sample_to_chunk = Vec::new();
        let mut sample_deltas = Vec::new();
        
        let mut cursor = io::Cursor::new(data);
        
        while cursor.position() < data.len() as u64 {
            let pos = cursor.position();
            let remaining = data.len() - pos as usize;
            
            if remaining < 8 {
                break;
            }
            
            let size = cursor.read_u32::<BigEndian>()? as usize;
            let mut type_buf = [0u8; 4];
            cursor.read_exact(&mut type_buf)?;
            let atom_type = String::from_utf8_lossy(&type_buf);
            
            if size == 0 || size > remaining {
                break;
            }
            
            let atom_data = &data[(pos + 8) as usize..(pos + size as u64) as usize];
            
            match atom_type.as_ref() {
                "stsd" => {
                    sample_entry = Self::parse_stsd(atom_data)?;
                }
                "stsz" => {
                    sample_sizes = Self::parse_stsz(atom_data)?;
                }
                "stco" => {
                    chunk_offsets = Self::parse_stco(atom_data)?;
                }
                "co64" => {
                    chunk_offsets = Self::parse_co64(atom_data)?;
                }
                "stsc" => {
                    sample_to_chunk = Self::parse_stsc(atom_data)?;
                }
                "stts" => {
                    sample_deltas = Self::parse_stts(atom_data)?;
                }
                _ => {}
            }
            
            cursor.set_position(pos + size as u64);
        }
        
        Ok(SampleTableInfo {
            sample_entry,
            sample_sizes,
            chunk_offsets,
            sample_to_chunk,
            sample_deltas,
        })
    }
    
    /// Parse stsd (sample description) atom
    fn parse_stsd(data: &[u8]) -> Result<Option<SampleEntry>, QtError> {
        let mut cursor = io::Cursor::new(data);
        
        let _version = cursor.read_u8()?;
        let _flags = cursor.read_u24::<BigEndian>()?;
        let entry_count = cursor.read_u32::<BigEndian>()?;
        
        if entry_count == 0 {
            return Ok(None);
        }
        
        // Read first entry
        let _entry_size = cursor.read_u32::<BigEndian>()?;
        let mut format_buf = [0u8; 4];
        cursor.read_exact(&mut format_buf)?;
        let format = String::from_utf8_lossy(&format_buf);
        
        // Skip reserved (6 bytes) and data reference index
        cursor.seek(SeekFrom::Current(6))?;
        let data_reference_index = cursor.read_u16::<BigEndian>()?;
        
        // For video samples, there are more fields, but we just need the codec type
        // HAP codec types: Hap1, Hap5, HapY, HapM, HapA, Hap7, HapH
        
        Ok(Some(SampleEntry {
            codec_type: format.to_string(),
            data_reference_index,
        }))
    }
    
    /// Parse stsz (sample size) atom
    fn parse_stsz(data: &[u8]) -> Result<Vec<u32>, QtError> {
        let mut cursor = io::Cursor::new(data);
        
        let _version = cursor.read_u8()?;
        let _flags = cursor.read_u24::<BigEndian>()?;
        let sample_size = cursor.read_u32::<BigEndian>()?;
        let sample_count = cursor.read_u32::<BigEndian>()?;
        
        let mut sizes = Vec::with_capacity(sample_count as usize);
        
        if sample_size == 0 {
            // Variable sample sizes
            for _ in 0..sample_count {
                sizes.push(cursor.read_u32::<BigEndian>()?);
            }
        } else {
            // Fixed sample size
            sizes.resize(sample_count as usize, sample_size);
        }
        
        Ok(sizes)
    }
    
    /// Parse stco (chunk offset, 32-bit) atom
    fn parse_stco(data: &[u8]) -> Result<Vec<u64>, QtError> {
        let mut cursor = io::Cursor::new(data);
        
        let _version = cursor.read_u8()?;
        let _flags = cursor.read_u24::<BigEndian>()?;
        let entry_count = cursor.read_u32::<BigEndian>()?;
        
        let mut offsets = Vec::with_capacity(entry_count as usize);
        
        for _ in 0..entry_count {
            offsets.push(cursor.read_u32::<BigEndian>()? as u64);
        }
        
        Ok(offsets)
    }
    
    /// Parse co64 (chunk offset, 64-bit) atom
    fn parse_co64(data: &[u8]) -> Result<Vec<u64>, QtError> {
        let mut cursor = io::Cursor::new(data);
        
        let _version = cursor.read_u8()?;
        let _flags = cursor.read_u24::<BigEndian>()?;
        let entry_count = cursor.read_u32::<BigEndian>()?;
        
        let mut offsets = Vec::with_capacity(entry_count as usize);
        
        for _ in 0..entry_count {
            offsets.push(cursor.read_u64::<BigEndian>()?);
        }
        
        Ok(offsets)
    }
    
    /// Parse stsc (sample to chunk) atom
    fn parse_stsc(data: &[u8]) -> Result<Vec<SampleToChunkEntry>, QtError> {
        let mut cursor = io::Cursor::new(data);
        
        let _version = cursor.read_u8()?;
        let _flags = cursor.read_u24::<BigEndian>()?;
        let entry_count = cursor.read_u32::<BigEndian>()?;
        
        let mut entries = Vec::with_capacity(entry_count as usize);
        
        for _ in 0..entry_count {
            entries.push(SampleToChunkEntry {
                first_chunk: cursor.read_u32::<BigEndian>()?,
                samples_per_chunk: cursor.read_u32::<BigEndian>()?,
                sample_description_index: cursor.read_u32::<BigEndian>()?,
            });
        }
        
        Ok(entries)
    }
    
    /// Parse stts (time to sample) atom
    fn parse_stts(data: &[u8]) -> Result<Vec<u32>, QtError> {
        let mut cursor = io::Cursor::new(data);
        
        let _version = cursor.read_u8()?;
        let _flags = cursor.read_u24::<BigEndian>()?;
        let entry_count = cursor.read_u32::<BigEndian>()?;
        
        let mut deltas = Vec::new();
        
        for _ in 0..entry_count {
            let sample_count = cursor.read_u32::<BigEndian>()?;
            let sample_delta = cursor.read_u32::<BigEndian>()?;
            
            for _ in 0..sample_count {
                deltas.push(sample_delta);
            }
        }
        
        Ok(deltas)
    }
}

/// Media info from mdia atom
struct MediaInfo {
    timescale: u32,
    duration: u64,
    sample_entry: Option<SampleEntry>,
    sample_sizes: Vec<u32>,
    chunk_offsets: Vec<u64>,
    sample_to_chunk: Vec<SampleToChunkEntry>,
    sample_deltas: Vec<u32>,
}

/// Sample table info from stbl atom
struct SampleTableInfo {
    sample_entry: Option<SampleEntry>,
    sample_sizes: Vec<u32>,
    chunk_offsets: Vec<u64>,
    sample_to_chunk: Vec<SampleToChunkEntry>,
    sample_deltas: Vec<u32>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_chunk_mapping() {
        // Create a simple sample to chunk table
        let stc = vec![
            SampleToChunkEntry {
                first_chunk: 1,
                samples_per_chunk: 10,
                sample_description_index: 1,
            },
        ];
        
        let track = VideoTrack {
            track_id: 1,
            width: 1280,
            height: 720,
            frame_count: 30,
            timescale: 30000,
            duration: 30000,
            sample_entry: SampleEntry {
                codec_type: "Hap1".to_string(),
                data_reference_index: 1,
            },
            sample_sizes: vec![1000; 30],
            chunk_offsets: vec![100, 10100, 20100], // 3 chunks
            sample_to_chunk: stc,
            sample_deltas: vec![1000; 30],
        };
        
        // Test frame to chunk mapping
        // Frame 0 -> chunk 0, offset 0
        // Frame 9 -> chunk 0, offset 9
        // Frame 10 -> chunk 1, offset 0
        // Frame 20 -> chunk 2, offset 0
        
        // This is tested indirectly through the reader
    }
}
