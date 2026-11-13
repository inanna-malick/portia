//! Validated half-open UTF-8 byte spans and extracted `todo!("...")` hole descriptors.

use std::ops::Range;

use portia_core::CoreProtocolError;
use serde::{Deserialize, Serialize};

/// Half-open UTF-8 byte span `[start..end)` enforcing `start <= end`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(try_from = "RawByteSpan", into = "RawByteSpan")]
pub struct ByteSpan {
    start: usize,
    end: usize,
}

#[derive(Serialize, Deserialize)]
struct RawByteSpan {
    start: usize,
    end: usize,
}

impl From<ByteSpan> for RawByteSpan {
    fn from(span: ByteSpan) -> Self {
        Self {
            start: span.start,
            end: span.end,
        }
    }
}

impl TryFrom<RawByteSpan> for ByteSpan {
    type Error = String;

    fn try_from(raw: RawByteSpan) -> Result<Self, Self::Error> {
        Self::try_new(raw.start, raw.end).map_err(|e| e.to_string())
    }
}

impl ByteSpan {
    /// Construct a validated half-open byte span `[start..end)`, rejecting `start > end`.
    pub fn try_new(start: usize, end: usize) -> Result<Self, CoreProtocolError> {
        if start > end {
            return Err(CoreProtocolError::BoundaryViolation(format!(
                "invalid ByteSpan [{start}..{end}): start > end"
            )));
        }
        Ok(Self { start, end })
    }

    /// Construct a half-open byte span `[start..end)`.
    ///
    /// # Panics
    ///
    /// Panics if `start > end`.
    #[inline]
    #[must_use]
    pub const fn new(start: usize, end: usize) -> Self {
        assert!(start <= end, "ByteSpan::new requires start <= end");
        Self { start, end }
    }

    /// Inclusive start byte offset.
    #[inline]
    #[must_use]
    pub const fn start(self) -> usize {
        self.start
    }

    /// Exclusive end byte offset.
    #[inline]
    #[must_use]
    pub const fn end(self) -> usize {
        self.end
    }

    /// Byte length `end - start`.
    #[inline]
    #[must_use]
    pub const fn len(self) -> usize {
        self.end - self.start
    }

    /// Return `true` when `start == end`.
    #[inline]
    #[must_use]
    pub const fn is_empty(self) -> bool {
        self.start == self.end
    }

    /// Convert into `std::ops::Range<usize>`.
    #[inline]
    #[must_use]
    pub const fn as_range(self) -> Range<usize> {
        self.start..self.end
    }

    /// Borrow `&source[self.start..self.end]` when both endpoints lie on valid UTF-8 char boundaries.
    #[inline]
    #[must_use]
    pub fn slice(self, source: &str) -> Option<&str> {
        source.get(self.start..self.end)
    }
}

impl From<ByteSpan> for Range<usize> {
    #[inline]
    fn from(span: ByteSpan) -> Self {
        span.as_range()
    }
}

impl TryFrom<Range<usize>> for ByteSpan {
    type Error = CoreProtocolError;

    #[inline]
    fn try_from(range: Range<usize>) -> Result<Self, Self::Error> {
        Self::try_new(range.start, range.end)
    }
}

impl From<ByteSpan> for (usize, usize) {
    #[inline]
    fn from(span: ByteSpan) -> Self {
        (span.start, span.end)
    }
}

impl TryFrom<(usize, usize)> for ByteSpan {
    type Error = CoreProtocolError;

    #[inline]
    fn try_from(pair: (usize, usize)) -> Result<Self, Self::Error> {
        Self::try_new(pair.0, pair.1)
    }
}

/// Extracted `todo!("...")` typed hole inside a Rust module.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExtractedRustTodoHole {
    pub hole_index: usize,
    pub fn_name: String,
    pub doc_comments: Vec<String>,
    pub signature: String,
    pub todo_macro_span: ByteSpan,
    pub fn_span: ByteSpan,
    pub intent_message: String,
}
