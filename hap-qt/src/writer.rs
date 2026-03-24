//! QuickTime HAP Writer
//!
//! Writes HAP video to QuickTime container format (.mov).
//! Implements the necessary atoms for a valid QuickTime file with HAP video.

use crate::frame_encoder::{HapEncodeError, HapFormat};
use byteorder::{BigEndian, WriteBytesExt};
use std::fs::File;
use std::io::{self, Seek, SeekFrom, Write};
use std::path::Path;
use thiserror::Error;

/// Errors that can occur during QuickTime writing
#[derive(Error, Debug)]
pub enum QtWriterError {
    #[error("IO error: {0}")]
    Io(#[from] io::Error),

    #[error("Invalid configuration: {0}")]
    InvalidConfig(String),

    #[error("HAP encoding error: {0}")]
    HapError(#[from] HapEncodeError),

    #[error("Writer already finalized")]
    AlreadyFinalized,

    #[error("No frames written")]
    NoFrames,
}

/// Video configuration for encoding
#[derive(Clone, Debug)]
pub struct VideoConfig {
    /// Frame width in pixels
    pub width: u32,
    /// Frame height in pixels
    pub height: u32,
    /// Frame rate (frames per second)
    pub fps: f32,
    /// HAP format variant
    pub codec: HapFormat,
    /// Timescale (time units per second, usually fps * 100)
    pub timescale: u32,
}

impl VideoConfig {
    /// Create a new video configuration with default timescale
    pub fn new(width: u32, height: u32, fps: f32, codec: HapFormat) -> Self {
        Self {
            width,
            height,
            fps,
            codec,
            timescale: (fps * 100.0) as u32,
        }
    }

    /// Calculate sample duration in timescale units
    pub fn sample_duration(&self) -> u32 {
        (self.timescale as f32 / self.fps).round() as u32
    }
}

/// Sample information for each frame
#[derive(Debug, Clone)]
struct SampleInfo {
    /// Size in bytes
    size: u32,
    /// Offset in file
    offset: u64,
    /// Duration in timescale units
    duration: u32,
}

/// QuickTime HAP Video Writer
///
/// Writes HAP-encoded frames to a QuickTime container.
/// Must call `finalize()` to complete the file.
pub struct QtHapWriter {
    /// Output file
    file: File,
    /// Video configuration
    config: VideoConfig,
    /// Track ID
    track_id: u32,
    /// Sample information for each frame
    samples: Vec<SampleInfo>,
    /// Current file position (for mdat data)
    mdat_position: u64,
    /// Whether the file has been finalized
    finalized: bool,
    /// MDAT start position (for calculating offsets)
    mdat_data_start: u64,
}

impl QtHapWriter {
    /// Create a new HAP video file
    ///
    /// # Arguments
    ///
    /// * `path` - Output file path
    /// * `config` - Video configuration
    ///
    /// # Example
    ///
    /// ```no_run
    /// use hap_qt::{QtHapWriter, VideoConfig, HapFormat};
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let config = VideoConfig::new(1920, 1080, 30.0, HapFormat::HapY);
    /// let mut writer = QtHapWriter::create("output.mov", config)?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn create<P: AsRef<Path>>(path: P, config: VideoConfig) -> Result<Self, QtWriterError> {
        let mut file = File::create(path)?;

        // Write ftyp atom (file type)
        Self::write_ftyp(&mut file)?;

        // Reserve space for moov atom (will be written at the end)
        // We'll write mdat first, then moov (modern fast-start format)

        // Write mdat atom header
        let _mdat_start = file.seek(SeekFrom::Current(0))?;
        // Write placeholder size (will update if needed, or use extended size)
        file.write_u32::<BigEndian>(1)?; // Extended size indicator
        file.write_all(b"mdat")?;
        file.write_u64::<BigEndian>(0)?; // Placeholder for extended size

        let mdat_data_start = file.seek(SeekFrom::Current(0))?;

        Ok(Self {
            file,
            config,
            track_id: 1,
            samples: Vec::new(),
            mdat_position: mdat_data_start,
            finalized: false,
            mdat_data_start,
        })
    }

    /// Write a HAP-encoded frame
    ///
    /// The frame data should be encoded using `HapFrameEncoder`.
    ///
    /// # Arguments
    ///
    /// * `hap_frame` - Encoded HAP frame data
    ///
    /// # Errors
    ///
    /// Returns an error if the writer has been finalized or writing fails
    pub fn write_frame(&mut self, hap_frame: &[u8]) -> Result<(), QtWriterError> {
        if self.finalized {
            return Err(QtWriterError::AlreadyFinalized);
        }

        let offset = self.mdat_position;
        let size = hap_frame.len() as u32;
        let duration = self.config.sample_duration();

        // Write frame data
        self.file.write_all(hap_frame)?;
        self.mdat_position += size as u64;

        // Record sample info
        self.samples.push(SampleInfo {
            size,
            offset,
            duration,
        });

        Ok(())
    }

    /// Get the number of frames written
    pub fn frame_count(&self) -> u32 {
        self.samples.len() as u32
    }
}

impl Drop for QtHapWriter {
    fn drop(&mut self) {
        // Try to finalize if not already finalized and we have frames
        if !self.finalized && !self.samples.is_empty() {
            // Mark as finalized first to prevent double attempts
            self.finalized = true;
            // We can't report errors in drop, but we can try to finalize
            let _ = self.try_finalize();
        }
    }
}

impl QtHapWriter {
    /// Try to finalize without consuming self (used by Drop)
    fn try_finalize(&mut self) -> Result<(), QtWriterError> {
        if self.samples.is_empty() {
            return Err(QtWriterError::NoFrames);
        }

        // Calculate mdat size
        let mdat_end = self.mdat_position;
        let mdat_data_size = mdat_end - self.mdat_data_start;
        let mdat_atom_size = mdat_data_size + 16;

        // Update mdat size
        self.file.seek(SeekFrom::Start(self.mdat_data_start - 16))?;
        self.file.write_u32::<BigEndian>(1)?;
        self.file.write_all(b"mdat")?;
        self.file.write_u64::<BigEndian>(mdat_atom_size as u64)?;

        // Seek to end to write moov
        self.file.seek(SeekFrom::End(0))?;

        // Write moov atom
        self.write_moov()?;

        // Ensure all data is flushed to disk
        self.file.flush()?;

        Ok(())
    }

    /// Finalize and write moov atom
    ///
    /// This must be called to complete a valid video file.
    /// After finalization, no more frames can be written.
    ///
    /// # Errors
    ///
    /// Returns an error if no frames were written or writing fails
    pub fn finalize(mut self) -> Result<(), QtWriterError> {
        if self.finalized {
            return Ok(());
        }

        if self.samples.is_empty() {
            return Err(QtWriterError::NoFrames);
        }

        // Mark as finalized first to prevent Drop from trying again
        self.finalized = true;

        // Calculate mdat size
        let mdat_end = self.mdat_position;
        let mdat_data_size = mdat_end - self.mdat_data_start;
        let mdat_atom_size = mdat_data_size + 16; // +16 for mdat header

        // Update mdat size at the beginning of mdat atom
        self.file.seek(SeekFrom::Start(self.mdat_data_start - 16))?;
        self.file.write_u32::<BigEndian>(1)?; // Extended size indicator
        self.file.write_all(b"mdat")?;
        self.file.write_u64::<BigEndian>(mdat_atom_size as u64)?;

        // Seek to end to write moov
        self.file.seek(SeekFrom::End(0))?;

        // Write moov atom
        self.write_moov()?;

        // Ensure all data is flushed to disk
        self.file.flush()?;

        Ok(())
    }

    /// Write ftyp atom (file type)
    fn write_ftyp(file: &mut File) -> io::Result<()> {
        // ftyp atom structure:
        // - Size (4 bytes)
        // - "ftyp" (4 bytes)
        // - Major brand "qt   " (4 bytes)
        // - Minor version (4 bytes)
        // - Compatible brands (variable)
        
        let brands: [&[u8]; 2] = [b"qt  ", b"qt  "];
        let size = 8 + 4 + 4 + (brands.len() * 4);

        file.write_u32::<BigEndian>(size as u32)?;
        file.write_all(b"ftyp")?;
        file.write_all(b"qt  ")?; // Major brand
        file.write_u32::<BigEndian>(0x00000200)?; // Minor version
        
        for brand in &brands {
            file.write_all(brand)?;
        }

        Ok(())
    }

    /// Write moov atom (movie metadata)
    fn write_moov(&mut self) -> io::Result<()> {
        // Build moov content
        let mut moov_content = Vec::new();

        // Write mvhd (movie header)
        self.write_mvhd(&mut moov_content)?;

        // Write trak (track)
        self.write_trak(&mut moov_content)?;

        // Write moov atom header + content
        let moov_size = 8 + moov_content.len();
        self.file.write_u32::<BigEndian>(moov_size as u32)?;
        self.file.write_all(b"moov")?;
        self.file.write_all(&moov_content)?;

        Ok(())
    }

    /// Write mvhd atom (movie header)
    fn write_mvhd(&self, writer: &mut dyn Write) -> io::Result<()> {
        let mut content = Vec::new();

        // Version (1 byte) - 0 for 32-bit times
        content.write_u8(0)?;
        // Flags (3 bytes)
        content.write_all(&[0, 0, 0])?;
        // Creation time (4 bytes) - 0
        content.write_u32::<BigEndian>(0)?;
        // Modification time (4 bytes) - 0
        content.write_u32::<BigEndian>(0)?;
        // Timescale (4 bytes)
        content.write_u32::<BigEndian>(self.config.timescale)?;
        // Duration (4 bytes)
        let total_duration = self.samples.iter().map(|s| s.duration as u64).sum::<u64>() as u32;
        content.write_u32::<BigEndian>(total_duration)?;
        // Preferred rate (4 bytes) - 1.0 (16.16 fixed point)
        content.write_u32::<BigEndian>(0x00010000)?;
        // Preferred volume (2 bytes) - 1.0 (8.8 fixed point)
        content.write_u16::<BigEndian>(0x0100)?;
        // Reserved (10 bytes)
        content.write_all(&[0; 10])?;
        // Matrix (36 bytes) - identity matrix
        let matrix = [
            0x00010000u32, 0, 0,
            0, 0x00010000u32, 0,
            0, 0, 0x40000000u32,
        ];
        for val in &matrix {
            content.write_u32::<BigEndian>(*val)?;
        }
        // Preview time (4 bytes) - 0
        content.write_u32::<BigEndian>(0)?;
        // Preview duration (4 bytes) - 0
        content.write_u32::<BigEndian>(0)?;
        // Poster time (4 bytes) - 0
        content.write_u32::<BigEndian>(0)?;
        // Selection time (4 bytes) - 0
        content.write_u32::<BigEndian>(0)?;
        // Selection duration (4 bytes) - 0
        content.write_u32::<BigEndian>(0)?;
        // Current time (4 bytes) - 0
        content.write_u32::<BigEndian>(0)?;
        // Next track ID (4 bytes)
        content.write_u32::<BigEndian>(self.track_id + 1)?;

        // Write mvhd atom
        writer.write_u32::<BigEndian>(8 + content.len() as u32)?;
        writer.write_all(b"mvhd")?;
        writer.write_all(&content)?;

        Ok(())
    }

    /// Write trak atom (track)
    fn write_trak(&self, writer: &mut dyn Write) -> io::Result<()> {
        let mut content = Vec::new();

        // Write tkhd (track header)
        self.write_tkhd(&mut content)?;

        // Write mdia (media)
        self.write_mdia(&mut content)?;

        // Write trak atom
        writer.write_u32::<BigEndian>(8 + content.len() as u32)?;
        writer.write_all(b"trak")?;
        writer.write_all(&content)?;

        Ok(())
    }

    /// Write tkhd atom (track header)
    fn write_tkhd(&self, writer: &mut dyn Write) -> io::Result<()> {
        let mut content = Vec::new();

        // Version (1 byte) - 0
        content.write_u8(0)?;
        // Flags (3 bytes) - 0x000007 (track enabled, in movie, in preview)
        content.write_all(&[0, 0, 0x0F])?;
        // Creation time (4 bytes)
        content.write_u32::<BigEndian>(0)?;
        // Modification time (4 bytes)
        content.write_u32::<BigEndian>(0)?;
        // Track ID (4 bytes)
        content.write_u32::<BigEndian>(self.track_id)?;
        // Reserved (4 bytes)
        content.write_u32::<BigEndian>(0)?;
        // Duration (4 bytes)
        let total_duration = self.samples.iter().map(|s| s.duration as u64).sum::<u64>() as u32;
        content.write_u32::<BigEndian>(total_duration)?;
        // Reserved (8 bytes)
        content.write_u64::<BigEndian>(0)?;
        // Layer (2 bytes) - 0
        content.write_u16::<BigEndian>(0)?;
        // Alternate group (2 bytes) - 0
        content.write_u16::<BigEndian>(0)?;
        // Volume (2 bytes) - 0 for video
        content.write_u16::<BigEndian>(0)?;
        // Reserved (2 bytes)
        content.write_u16::<BigEndian>(0)?;
        // Matrix (36 bytes) - identity
        let matrix = [
            0x00010000u32, 0, 0,
            0, 0x00010000u32, 0,
            0, 0, 0x40000000u32,
        ];
        for val in &matrix {
            content.write_u32::<BigEndian>(*val)?;
        }
        // Width (4 bytes) - 16.16 fixed point
        content.write_u32::<BigEndian>(self.config.width << 16)?;
        // Height (4 bytes) - 16.16 fixed point
        content.write_u32::<BigEndian>(self.config.height << 16)?;

        // Write tkhd atom
        writer.write_u32::<BigEndian>(8 + content.len() as u32)?;
        writer.write_all(b"tkhd")?;
        writer.write_all(&content)?;

        Ok(())
    }

    /// Write mdia atom (media)
    fn write_mdia(&self, writer: &mut dyn Write) -> io::Result<()> {
        let mut content = Vec::new();

        // Write mdhd (media header)
        self.write_mdhd(&mut content)?;

        // Write hdlr (handler)
        self.write_hdlr(&mut content)?;

        // Write minf (media info)
        self.write_minf(&mut content)?;

        // Write mdia atom
        writer.write_u32::<BigEndian>(8 + content.len() as u32)?;
        writer.write_all(b"mdia")?;
        writer.write_all(&content)?;

        Ok(())
    }

    /// Write mdhd atom (media header)
    fn write_mdhd(&self, writer: &mut dyn Write) -> io::Result<()> {
        let mut content = Vec::new();

        // Version (1 byte) - 0
        content.write_u8(0)?;
        // Flags (3 bytes)
        content.write_all(&[0, 0, 0])?;
        // Creation time (4 bytes)
        content.write_u32::<BigEndian>(0)?;
        // Modification time (4 bytes)
        content.write_u32::<BigEndian>(0)?;
        // Timescale (4 bytes)
        content.write_u32::<BigEndian>(self.config.timescale)?;
        // Duration (4 bytes)
        let total_duration = self.samples.iter().map(|s| s.duration as u64).sum::<u64>() as u32;
        content.write_u32::<BigEndian>(total_duration)?;
        // Language (2 bytes) - 0 (English)
        content.write_u16::<BigEndian>(0)?;
        // Predefined (2 bytes) - 0
        content.write_u16::<BigEndian>(0)?;

        // Write mdhd atom
        writer.write_u32::<BigEndian>(8 + content.len() as u32)?;
        writer.write_all(b"mdhd")?;
        writer.write_all(&content)?;

        Ok(())
    }

    /// Write hdlr atom (handler)
    fn write_hdlr(&self, writer: &mut dyn Write) -> io::Result<()> {
        let mut content = Vec::new();

        // Version (1 byte) - 0
        content.write_u8(0)?;
        // Flags (3 bytes)
        content.write_all(&[0, 0, 0])?;
        // Predefined (4 bytes) - 0
        content.write_u32::<BigEndian>(0)?;
        // Handler type (4 bytes) - "vide" for video
        content.write_all(b"vide")?;
        // Reserved (12 bytes)
        content.write_all(&[0; 12])?;
        // Component name (null-terminated string) - empty
        content.write_u8(0)?;

        // Write hdlr atom
        writer.write_u32::<BigEndian>(8 + content.len() as u32)?;
        writer.write_all(b"hdlr")?;
        writer.write_all(&content)?;

        Ok(())
    }

    /// Write minf atom (media info)
    fn write_minf(&self, writer: &mut dyn Write) -> io::Result<()> {
        let mut content = Vec::new();

        // Write vmhd (video media header)
        self.write_vmhd(&mut content)?;

        // Write dinf (data info)
        self.write_dinf(&mut content)?;

        // Write stbl (sample table)
        self.write_stbl(&mut content)?;

        // Write minf atom
        writer.write_u32::<BigEndian>(8 + content.len() as u32)?;
        writer.write_all(b"minf")?;
        writer.write_all(&content)?;

        Ok(())
    }

    /// Write vmhd atom (video media header)
    fn write_vmhd(&self, writer: &mut dyn Write) -> io::Result<()> {
        let mut content = Vec::new();

        // Version (1 byte) - 0
        content.write_u8(0)?;
        // Flags (3 bytes) - 1
        content.write_all(&[0, 0, 1])?;
        // Graphics mode (2 bytes) - 0 (copy)
        content.write_u16::<BigEndian>(0)?;
        // Opcolor (6 bytes) - 0, 0, 0
        content.write_all(&[0; 6])?;

        // Write vmhd atom
        writer.write_u32::<BigEndian>(8 + content.len() as u32)?;
        writer.write_all(b"vmhd")?;
        writer.write_all(&content)?;

        Ok(())
    }

    /// Write dinf atom (data info)
    fn write_dinf(&self, writer: &mut dyn Write) -> io::Result<()> {
        let mut content = Vec::new();

        // Write dref (data reference)
        self.write_dref(&mut content)?;

        // Write dinf atom
        writer.write_u32::<BigEndian>(8 + content.len() as u32)?;
        writer.write_all(b"dinf")?;
        writer.write_all(&content)?;

        Ok(())
    }

    /// Write dref atom (data reference)
    fn write_dref(&self, writer: &mut dyn Write) -> io::Result<()> {
        let mut content = Vec::new();

        // Version (1 byte) - 0
        content.write_u8(0)?;
        // Flags (3 bytes)
        content.write_all(&[0, 0, 0])?;
        // Entry count (4 bytes) - 1
        content.write_u32::<BigEndian>(1)?;

        // Self-contained data reference entry (url)
        // Size (4 bytes)
        content.write_u32::<BigEndian>(12)?;
        // Type (4 bytes) - "url "
        content.write_all(b"url ")?;
        // Version/flags (4 bytes) - 1 (self-contained)
        content.write_u32::<BigEndian>(1)?;

        // Write dref atom
        writer.write_u32::<BigEndian>(8 + content.len() as u32)?;
        writer.write_all(b"dref")?;
        writer.write_all(&content)?;

        Ok(())
    }

    /// Write stbl atom (sample table)
    fn write_stbl(&self, writer: &mut dyn Write) -> io::Result<()> {
        let mut content = Vec::new();

        // Write stsd (sample description) - HAP specific
        self.write_stsd(&mut content)?;

        // Write stts (time to sample)
        self.write_stts(&mut content)?;

        // Write stsc (sample to chunk)
        self.write_stsc(&mut content)?;

        // Write stsz (sample size)
        self.write_stsz(&mut content)?;

        // Write stco (chunk offset)
        self.write_stco(&mut content)?;

        // Write stbl atom
        writer.write_u32::<BigEndian>(8 + content.len() as u32)?;
        writer.write_all(b"stbl")?;
        writer.write_all(&content)?;

        Ok(())
    }

    /// Write stsd atom (sample description)
    fn write_stsd(&self, writer: &mut dyn Write) -> io::Result<()> {
        let mut content = Vec::new();

        // Version (1 byte) - 0
        content.write_u8(0)?;
        // Flags (3 bytes)
        content.write_all(&[0, 0, 0])?;
        // Entry count (4 bytes) - 1
        content.write_u32::<BigEndian>(1)?;

        // Video sample entry
        self.write_video_sample_entry(&mut content)?;

        // Write stsd atom
        writer.write_u32::<BigEndian>(8 + content.len() as u32)?;
        writer.write_all(b"stsd")?;
        writer.write_all(&content)?;

        Ok(())
    }

    /// Write video sample entry for HAP
    fn write_video_sample_entry(&self, writer: &mut dyn Write) -> io::Result<()> {
        let mut content = Vec::new();

        // Sample entry base:
        // Reserved (6 bytes)
        content.write_all(&[0; 6])?;
        // Data reference index (2 bytes) - 1
        content.write_u16::<BigEndian>(1)?;

        // Visual sample entry:
        // Predefined (2 bytes) - 0
        content.write_u16::<BigEndian>(0)?;
        // Reserved (2 bytes) - 0
        content.write_u16::<BigEndian>(0)?;
        // Predefined (3 * 4 bytes) - 0
        content.write_u32::<BigEndian>(0)?;
        content.write_u32::<BigEndian>(0)?;
        content.write_u32::<BigEndian>(0)?;
        // Width (2 bytes)
        content.write_u16::<BigEndian>(self.config.width as u16)?;
        // Height (2 bytes)
        content.write_u16::<BigEndian>(self.config.height as u16)?;
        // Horizontal resolution (4 bytes) - 72 dpi (16.16 fixed point)
        content.write_u32::<BigEndian>(0x00480000)?;
        // Vertical resolution (4 bytes) - 72 dpi
        content.write_u32::<BigEndian>(0x00480000)?;
        // Reserved (4 bytes) - 0
        content.write_u32::<BigEndian>(0)?;
        // Frame count (2 bytes) - 1
        content.write_u16::<BigEndian>(1)?;

        // Compressor name (32 bytes) - null-terminated string
        let codec_name = self.config.codec.codec_name();
        let mut name_bytes = vec![0u8; 32];
        name_bytes[0] = codec_name.len() as u8;
        name_bytes[1..1 + codec_name.len()].copy_from_slice(codec_name.as_bytes());
        content.write_all(&name_bytes)?;

        // Depth (2 bytes) - 24
        content.write_u16::<BigEndian>(24)?;
        // Predefined (2 bytes) - -1 (0xFFFF)
        content.write_i16::<BigEndian>(-1)?;

        // Write sample entry
        let codec_fourcc = self.config.codec.codec_name().as_bytes();
        writer.write_u32::<BigEndian>(8 + content.len() as u32)?;
        writer.write_all(codec_fourcc)?;
        writer.write_all(&content)?;

        Ok(())
    }

    /// Write stts atom (time to sample)
    fn write_stts(&self, writer: &mut dyn Write) -> io::Result<()> {
        let mut content = Vec::new();

        // Version (1 byte) - 0
        content.write_u8(0)?;
        // Flags (3 bytes)
        content.write_all(&[0, 0, 0])?;
        // Entry count (4 bytes)
        content.write_u32::<BigEndian>(1)?;
        // Sample count (4 bytes)
        content.write_u32::<BigEndian>(self.samples.len() as u32)?;
        // Sample duration (4 bytes)
        content.write_u32::<BigEndian>(self.config.sample_duration())?;

        // Write stts atom
        writer.write_u32::<BigEndian>(8 + content.len() as u32)?;
        writer.write_all(b"stts")?;
        writer.write_all(&content)?;

        Ok(())
    }

    /// Write stsc atom (sample to chunk)
    fn write_stsc(&self, writer: &mut dyn Write) -> io::Result<()> {
        let mut content = Vec::new();

        // Version (1 byte) - 0
        content.write_u8(0)?;
        // Flags (3 bytes)
        content.write_all(&[0, 0, 0])?;
        // Entry count (4 bytes) - 1 (one chunk per sample for simplicity)
        content.write_u32::<BigEndian>(1)?;
        // First chunk (4 bytes) - 1
        content.write_u32::<BigEndian>(1)?;
        // Samples per chunk (4 bytes)
        content.write_u32::<BigEndian>(1)?;
        // Sample description index (4 bytes) - 1
        content.write_u32::<BigEndian>(1)?;

        // Write stsc atom
        writer.write_u32::<BigEndian>(8 + content.len() as u32)?;
        writer.write_all(b"stsc")?;
        writer.write_all(&content)?;

        Ok(())
    }

    /// Write stsz atom (sample size)
    fn write_stsz(&self, writer: &mut dyn Write) -> io::Result<()> {
        let mut content = Vec::new();

        // Version (1 byte) - 0
        content.write_u8(0)?;
        // Flags (3 bytes)
        content.write_all(&[0, 0, 0])?;
        // Sample size (4 bytes) - 0 (variable size)
        content.write_u32::<BigEndian>(0)?;
        // Sample count (4 bytes)
        content.write_u32::<BigEndian>(self.samples.len() as u32)?;

        // Sample sizes
        for sample in &self.samples {
            content.write_u32::<BigEndian>(sample.size)?;
        }

        // Write stsz atom
        writer.write_u32::<BigEndian>(8 + content.len() as u32)?;
        writer.write_all(b"stsz")?;
        writer.write_all(&content)?;

        Ok(())
    }

    /// Write stco atom (chunk offset, 32-bit)
    fn write_stco(&self, writer: &mut dyn Write) -> io::Result<()> {
        let mut content = Vec::new();

        // Version (1 byte) - 0
        content.write_u8(0)?;
        // Flags (3 bytes)
        content.write_all(&[0, 0, 0])?;
        // Entry count (4 bytes)
        content.write_u32::<BigEndian>(self.samples.len() as u32)?;

        // Chunk offsets (absolute file offsets)
        for sample in &self.samples {
            // Offset is absolute position in file
            content.write_u32::<BigEndian>(sample.offset as u32)?;
        }

        // Write stco atom
        writer.write_u32::<BigEndian>(8 + content.len() as u32)?;
        writer.write_all(b"stco")?;
        writer.write_all(&content)?;

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Read;

    #[test]
    fn test_video_config() {
        let config = VideoConfig::new(1920, 1080, 30.0, HapFormat::HapY);
        assert_eq!(config.width, 1920);
        assert_eq!(config.height, 1080);
        assert_eq!(config.fps, 30.0);
        assert_eq!(config.timescale, 3000);
        assert_eq!(config.sample_duration(), 100);
    }

    #[test]
    fn test_write_hap_file() {
        let temp_dir = std::env::temp_dir();
        let output_path = temp_dir.join("test_hap_writer.mov");

        {
            let config = VideoConfig::new(64, 64, 30.0, HapFormat::Hap1);
            let mut writer = QtHapWriter::create(&output_path, config).unwrap();

            // Write a few dummy frames (not valid HAP, but tests structure)
            for i in 0..5 {
                let mut frame_data = vec![0u8; 100];
                frame_data[0] = 0xAB;
                frame_data[4] = i as u8;
                writer.write_frame(&frame_data).unwrap();
            }

            assert_eq!(writer.frame_count(), 5);

            writer.finalize().unwrap();
        }

        // Verify file was created
        assert!(output_path.exists());

        // Read and verify basic structure
        let mut file = File::open(&output_path).unwrap();
        let mut data = Vec::new();
        file.read_to_end(&mut data).unwrap();

        // Check ftyp signature
        assert_eq!(&data[4..8], b"ftyp");
        assert_eq!(&data[8..12], b"qt  ");

        // Check mdat atom exists
        let mdat_pos = data.windows(4).position(|w| w == b"mdat").unwrap();
        assert!(mdat_pos > 0);

        // Check moov atom exists at end
        let moov_pos = data.windows(4).rposition(|w| w == b"moov").unwrap();
        assert!(moov_pos > 0);

        // Cleanup
        let _ = std::fs::remove_file(&output_path);
    }

    #[test]
    fn test_write_no_frames() {
        let temp_dir = std::env::temp_dir();
        let output_path = temp_dir.join("test_empty.mov");

        let config = VideoConfig::new(64, 64, 30.0, HapFormat::Hap1);
        let writer = QtHapWriter::create(&output_path, config).unwrap();

        // Try to finalize without writing frames
        let result = writer.finalize();
        assert!(matches!(result, Err(QtWriterError::NoFrames)));

        // Cleanup if file exists
        let _ = std::fs::remove_file(&output_path);
    }

    #[test]
    fn test_ftyp_atom_size() {
        let temp_dir = std::env::temp_dir();
        let output_path = temp_dir.join("test_ftyp_size.mov");

        let config = VideoConfig::new(64, 64, 30.0, HapFormat::Hap1);
        let mut writer = QtHapWriter::create(&output_path, config).unwrap();
        writer.write_frame(&vec![0xABu8; 100]).unwrap();
        writer.finalize().unwrap();

        let mut file = File::open(&output_path).unwrap();
        let mut data = Vec::new();
        file.read_to_end(&mut data).unwrap();

        // ftyp declared size must match actual content
        let ftyp_size = u32::from_be_bytes([data[0], data[1], data[2], data[3]]) as usize;
        assert_eq!(&data[4..8], b"ftyp");
        // The next atom must start exactly at ftyp_size offset
        let next_atom_type = &data[ftyp_size + 4..ftyp_size + 8];
        assert_eq!(next_atom_type, b"mdat", "atom after ftyp must be mdat");

        let _ = std::fs::remove_file(&output_path);
    }

    #[test]
    #[cfg(feature = "cpu-compression")]
    fn test_roundtrip_encode_decode() {
        use crate::{QtHapReader, HapFrameEncoder, CompressionMode};

        let temp_dir = std::env::temp_dir();
        let output_path = temp_dir.join("test_roundtrip.mov");

        let width = 64u32;
        let height = 64u32;
        let frame_count = 5u32;

        // Encode
        {
            let config = VideoConfig::new(width, height, 30.0, HapFormat::Hap1);
            let mut encoder = HapFrameEncoder::new(HapFormat::Hap1, width, height).unwrap();
            encoder.set_compression(CompressionMode::Snappy);
            let mut writer = QtHapWriter::create(&output_path, config).unwrap();

            for _ in 0..frame_count {
                let rgba = vec![128u8; (width * height * 4) as usize];
                let hap_frame = encoder.encode(&rgba).unwrap();
                writer.write_frame(&hap_frame).unwrap();
            }

            writer.finalize().unwrap();
        }

        // Decode and verify
        {
            let mut reader = QtHapReader::open(&output_path).unwrap();
            assert_eq!(reader.resolution(), (width, height));
            assert_eq!(reader.frame_count(), frame_count);

            // Read every frame
            for i in 0..frame_count {
                let frame = reader.read_frame(i).unwrap();
                assert!(!frame.texture_data.is_empty());
            }
        }

        let _ = std::fs::remove_file(&output_path);
    }
}
