//! Header parsing and bounded mappings of immutable checkpoint files.

use crate::SurgeonError;
use memmap2::{Mmap, MmapOptions};
use safetensors::tensor::Metadata;
use std::fs::File;
use std::io::{Read, Write};
use std::path::Path;

const WINDOW_BYTES: usize = 8 * 1024 * 1024;
const MAX_HEADER_BYTES: u64 = 100_000_000;

fn io(path: &Path, source: std::io::Error) -> SurgeonError {
    SurgeonError::Io {
        path: path.to_owned(),
        source,
    }
}

fn format(path: &Path, message: impl Into<String>) -> SurgeonError {
    SurgeonError::SafetensorsFormat {
        path: path.to_owned(),
        message: message.into(),
    }
}

/// Parse only the header, retaining the official metadata validation rules.
pub(crate) fn read_metadata(path: &Path) -> Result<(u64, Metadata), SurgeonError> {
    let mut file = File::open(path).map_err(|error| io(path, error))?;
    let length = file.metadata().map_err(|error| io(path, error))?.len();
    if length < 8 {
        return Err(format(
            path,
            "file is shorter than the header length prefix",
        ));
    }
    let mut prefix = [0; 8];
    file.read_exact(&mut prefix)
        .map_err(|error| io(path, error))?;
    let header_length = u64::from_le_bytes(prefix);
    if header_length > MAX_HEADER_BYTES {
        return Err(format(path, "safetensors header exceeds 100,000,000 bytes"));
    }
    let data_start = header_length + 8;
    if data_start > length {
        return Err(format(path, "header extends beyond the end of the file"));
    }
    let header_length = usize::try_from(header_length)
        .map_err(|_| format(path, "header length does not fit in usize"))?;
    let mut header = vec![0; header_length];
    file.read_exact(&mut header)
        .map_err(|error| io(path, error))?;
    let metadata: Metadata =
        serde_json::from_slice(&header).map_err(|error| format(path, error.to_string()))?;
    let data_length = u64::try_from(metadata.data_len())
        .map_err(|_| format(path, "tensor data length does not fit in u64"))?;
    if data_start.checked_add(data_length) != Some(length) {
        return Err(format(
            path,
            "tensor offsets do not cover the complete file payload",
        ));
    }
    Ok((data_start, metadata))
}

/// The caller must keep the input checkpoint immutable while this reader exists.
pub(crate) struct Reader<'a> {
    file: &'a File,
    path: &'a Path,
    length: u64,
    mapping: Option<Mmap>,
    start: u64,
}

impl<'a> Reader<'a> {
    pub(crate) fn path(&self) -> &'a Path {
        self.path
    }

    pub(crate) fn new(file: &'a File, path: &'a Path) -> Result<Self, SurgeonError> {
        let length = file.metadata().map_err(|error| io(path, error))?.len();
        Ok(Self {
            file,
            path,
            length,
            mapping: None,
            start: 0,
        })
    }

    fn chunk(&mut self, offset: u64, length: usize) -> Result<&[u8], SurgeonError> {
        let requested = u64::try_from(length)
            .map_err(|_| format(self.path, "read length does not fit in u64"))?;
        if offset
            .checked_add(requested)
            .is_none_or(|end| end > self.length)
        {
            return Err(format(self.path, "read extends beyond the end of the file"));
        }
        if length == 0 {
            return Ok(&[]);
        }
        let contains = self.mapping.as_ref().is_some_and(|mapping| {
            offset >= self.start && offset - self.start < mapping.len() as u64
        });
        if !contains {
            // Drop the previous window before mapping another to bound resident pages.
            self.mapping = None;
            let length = (self.length - offset).min(WINDOW_BYTES as u64) as usize;
            // SAFETY: Only a valid, nonempty range within the open file is mapped.
            // The public operation requires checkpoint inputs to remain immutable;
            // this crate never writes input files. No reference outlives its mapping.
            let mapping = unsafe { MmapOptions::new().offset(offset).len(length).map(self.file) }
                .map_err(|error| io(self.path, error))?;
            self.start = offset;
            self.mapping = Some(mapping);
        }
        let mapping = self
            .mapping
            .as_ref()
            .expect("a nonempty read has a mapping");
        let start = (offset - self.start) as usize;
        let end = start + length.min(mapping.len() - start);
        Ok(&mapping[start..end])
    }

    pub(crate) fn read_exact(
        &mut self,
        mut offset: u64,
        mut bytes: &mut [u8],
    ) -> Result<(), SurgeonError> {
        while !bytes.is_empty() {
            let chunk = self.chunk(offset, bytes.len())?;
            let length = chunk.len();
            bytes[..length].copy_from_slice(chunk);
            bytes = &mut bytes[length..];
            offset += length as u64;
        }
        Ok(())
    }

    pub(crate) fn copy_range(
        &mut self,
        mut offset: u64,
        mut length: usize,
        writer: &mut dyn Write,
    ) -> Result<(), SurgeonError> {
        while length != 0 {
            let bytes = self.chunk(offset, length)?;
            let copied = bytes.len();
            let result = writer.write_all(bytes);
            result.map_err(|error| io(self.path, error))?;
            length -= copied;
            offset += copied as u64;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::{Seek, SeekFrom};

    #[test]
    fn sequential_reads_reuse_the_window_until_its_boundary() {
        let file = tempfile::NamedTempFile::new().unwrap();
        file.as_file()
            .set_len((2 * WINDOW_BYTES + 32) as u64)
            .unwrap();
        let mut reader = Reader::new(file.as_file(), file.path()).unwrap();
        let mut bytes = [1; 16];
        reader.read_exact(0, &mut bytes).unwrap();
        let original = reader.mapping.as_ref().unwrap().as_ptr();
        for offset in [16, 4096, WINDOW_BYTES - bytes.len()] {
            reader.read_exact(offset as u64, &mut bytes).unwrap();
            assert_eq!(reader.start, 0);
            assert_eq!(reader.mapping.as_ref().unwrap().as_ptr(), original);
            assert_eq!(reader.mapping.as_ref().unwrap().len(), WINDOW_BYTES);
            assert_eq!(bytes, [0; 16]);
        }
        reader
            .read_exact((WINDOW_BYTES - 8) as u64, &mut bytes)
            .unwrap();
        assert_eq!(reader.start, WINDOW_BYTES as u64);
        assert_eq!(reader.mapping.as_ref().unwrap().len(), WINDOW_BYTES);
        assert_eq!(bytes, [0; 16]);
        reader
            .read_exact((2 * WINDOW_BYTES) as u64, &mut bytes)
            .unwrap();
        assert_eq!(reader.start, (2 * WINDOW_BYTES) as u64);
        assert_eq!(reader.mapping.as_ref().unwrap().len(), 32);
    }

    #[test]
    fn reads_and_copies_across_mapping_boundaries() {
        let mut file = tempfile::NamedTempFile::new().unwrap();
        file.as_file().set_len((WINDOW_BYTES + 32) as u64).unwrap();
        let start = (WINDOW_BYTES - 3) as u64;
        file.seek(SeekFrom::Start(start)).unwrap();
        let expected: Vec<u8> = (0..32).collect();
        file.write_all(&expected).unwrap();
        let mut reader = Reader::new(file.as_file(), file.path()).unwrap();
        reader.read_exact(0, &mut [0]).unwrap();
        let mut actual = [0; 32];
        reader.read_exact(start, &mut actual).unwrap();
        assert_eq!(actual.as_slice(), expected);
        let mut copied = Vec::new();
        reader
            .copy_range(start, expected.len(), &mut copied)
            .unwrap();
        assert_eq!(copied, expected);
        assert!(reader.mapping.as_ref().unwrap().len() <= WINDOW_BYTES);
        assert!(reader.read_exact(u64::MAX, &mut [0]).is_err());
        assert!(reader
            .copy_range((WINDOW_BYTES + 32) as u64, 1, &mut copied)
            .is_err());
    }

    #[test]
    fn metadata_validation_rejects_incomplete_and_trailing_payloads() {
        let mut file = tempfile::NamedTempFile::new().unwrap();
        let header = br#"{"x":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}"#;
        file.write_all(&(header.len() as u64).to_le_bytes())
            .unwrap();
        file.write_all(header).unwrap();
        file.write_all(&[0; 4]).unwrap();
        let (data_start, metadata) = read_metadata(file.path()).unwrap();
        assert_eq!(data_start, 8 + header.len() as u64);
        assert_eq!(metadata.data_len(), 4);
        let bytes = std::fs::read(file.path()).unwrap();
        safetensors::SafeTensors::deserialize(&bytes).unwrap();
        file.write_all(&[0]).unwrap();
        assert!(read_metadata(file.path()).is_err());
        file.as_file().set_len(data_start + 3).unwrap();
        assert!(read_metadata(file.path()).is_err());
        file.as_file().set_len(7).unwrap();
        assert!(read_metadata(file.path()).is_err());
    }
}
