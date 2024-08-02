#![feature(assert_matches)]

use std::cmp::{min, Ordering};
use std::io;
use std::io::{Read, Write};
use std::os::fd::AsRawFd;
use std::ptr::{slice_from_raw_parts, slice_from_raw_parts_mut};
#[cfg(any(feature = "futures", feature = "tokio"))]
use std::task::{Context, Poll, Waker};

use madvise::AccessPattern;
pub use mmap_rs::{MmapFlags, MmapMut, MmapOptions, PageSize, PageSizes, UnsafeMmapFlags};
use thiserror::Error;
use tracing::{debug, trace};

#[derive(Debug, Error)]
pub enum QuantumRingError {
    #[error("allowed page sizes are not supported")]
    UnsupportedPageSizes {
        requested: PageSizes,
        supported: PageSizes,
    },

    #[error("mmap-rs error: {:?}", 0)]
    Mmap(#[from] mmap_rs::Error),

    #[error("io error: {:?}", 0)]
    Io(#[from] std::io::Error),
}

#[derive(Debug, Copy, Clone)]
struct PageSizeIter {
    i: usize,
    page_sizes: PageSizes,
}

impl Iterator for PageSizeIter {
    type Item = PageSize;

    fn next(&mut self) -> Option<Self::Item> {
        let bits = self.page_sizes.bits();
        let mut next = None;
        while next.is_none() && self.i < std::mem::size_of_val(&bits) * 8 {
            if 1usize << self.i & bits > 0 {
                next = Some(PageSize(self.i));
            }
            self.i += 1;
        }
        next
    }
}

impl PageSizeIter {
    pub fn new(page_sizes: PageSizes) -> Self {
        Self { i: 0, page_sizes }
    }
}

pub struct QuantumRing {
    pub(crate) first_map: MmapMut,
    #[allow(unused)]
    pub(crate) second_map: MmapMut,
    #[allow(unused)]
    pub(crate) padding_map: Option<MmapMut>,

    pub(crate) read: usize,
    pub(crate) write: usize,
    pub(crate) len: usize,
    pub(crate) capacity: usize,

    #[cfg(any(feature = "futures", feature = "tokio"))]
    pub(crate) read_waker: Option<Waker>,
    #[cfg(any(feature = "futures", feature = "tokio"))]
    pub(crate) write_waker: Option<Waker>,
}

impl QuantumRing {
    pub fn new(
        pages: usize,
        page_size: PageSize,
        min_padding: usize,
    ) -> Result<Self, QuantumRingError> {
        let supported_page_sizes = MmapOptions::page_sizes()?;
        {
            let sizes = PageSizes::from_bits(1usize << page_size.0).unwrap();
            if !supported_page_sizes.contains(sizes) {
                return Err(QuantumRingError::UnsupportedPageSizes {
                    requested: sizes,
                    supported: supported_page_sizes,
                });
            }
        }

        let (first, second, padding) = unsafe {
            let page_bytes = 1usize << page_size.0;
            let physical_size = pages * page_bytes;

            let padding_pages = (min_padding + page_bytes - 1) / page_bytes; // round up
            let padding_bytes = padding_pages * page_bytes;

            let reserved_size = (2 * physical_size) + padding_bytes;
            let mut reserved = MmapOptions::new(reserved_size)?.map_none()?;

            let first_reserved = reserved.split_to(physical_size)?;
            let (second_reserved, padding_reserved) = match padding_pages {
                0 => (reserved, None),
                _ => (reserved.split_to(physical_size)?, Some(reserved)),
            };

            trace!(
                first_reserved_start = first_reserved.start(),
                first_reserved_size = first_reserved.size(),
                second_reserved_start = second_reserved.start(),
                second_reserved_size = second_reserved.size(),
                padding_reserved_start = padding_reserved.as_ref().map(|p| p.start()).unwrap_or_default(),
                padding_reserved_size = padding_reserved.as_ref().map(|p| p.size()).unwrap_or_default(),
                "reserved 2 * {physical_size} + {padding_bytes} = {reserved_size} in virtual memory space"
            );

            let mut fd = mkstemp::TempFile::new(
                std::env::temp_dir()
                    .join("quantum-ring-XXXXXX")
                    .to_str()
                    .unwrap(),
                true,
            )?;
            fd.inner().set_len(physical_size as u64)?;
            let raw_fd = fd.inner().as_raw_fd();
            trace!(
                fd = raw_fd,
                file = fd.path(),
                "tmp file created: {}",
                fd.path()
            );

            let first = MmapOptions::new(first_reserved.size())?
                .with_address(first_reserved.start())
                .with_flags(MmapFlags::SHARED | MmapFlags::TRANSPARENT_HUGE_PAGES)
                .with_unsafe_flags(UnsafeMmapFlags::MAP_FIXED)
                .with_file(fd.inner(), 0)
                .map_mut()?;
            let second = MmapOptions::new(second_reserved.size())?
                .with_address(second_reserved.start())
                .with_flags(MmapFlags::SHARED | MmapFlags::TRANSPARENT_HUGE_PAGES)
                .with_unsafe_flags(UnsafeMmapFlags::MAP_FIXED)
                .with_file(fd.inner(), 0)
                .map_mut()?;
            let padding = if let Some(padding_reserved) = padding_reserved.as_ref() {
                Some(
                    MmapOptions::new(padding_reserved.size())?
                        .with_address(padding_reserved.start())
                        .with_flags(MmapFlags::SHARED | MmapFlags::TRANSPARENT_HUGE_PAGES)
                        .with_unsafe_flags(UnsafeMmapFlags::MAP_FIXED)
                        .with_file(fd.inner(), 0)
                        .map_mut()?,
                )
            } else {
                None
            };

            std::mem::forget(first_reserved);
            std::mem::forget(second_reserved);
            std::mem::forget(padding_reserved);

            (first, second, padding)
        };

        debug!(
            first_map_start = first.start(),
            first_map_size = first.size(),
            second_map_start = second.start(),
            second_map_size = second.size(),
            padding_map_start = padding.as_ref().map(|p| p.start()).unwrap_or_default(),
            padding_map_size = padding.as_ref().map(|p| p.size()).unwrap_or_default(),
            "QuantumRing at {:x}, size: {}, wraparound at {:x}, size: {}, padding at {:x}, padding size: {}",
            first.start(),
            first.size(),
            second.start(),
            second.size(),
            padding.as_ref().map(|p| p.start()).unwrap_or_default(),
            padding.as_ref().map(|p| p.size()).unwrap_or_default(),
        );

        let qr = Self {
            read: 0,
            write: 0,
            len: 0,
            capacity: first.size(),
            first_map: first,
            second_map: second,
            padding_map: padding,

            #[cfg(any(feature = "futures", feature = "tokio"))]
            read_waker: None,
            #[cfg(any(feature = "futures", feature = "tokio"))]
            write_waker: None,
        };
        qr.madvise(AccessPattern::Sequential)?;
        Ok(qr)
    }

    pub fn new_with_size(
        size: usize,
        allowed_page_sizes: PageSizes,
        min_padding: usize,
    ) -> Result<Self, QuantumRingError> {
        let supported_page_sizes = MmapOptions::page_sizes()?;
        if !supported_page_sizes.intersects(allowed_page_sizes) {
            return Err(QuantumRingError::UnsupportedPageSizes {
                requested: allowed_page_sizes,
                supported: supported_page_sizes,
            });
        }

        let page_configs = PageSizeIter::new(supported_page_sizes & allowed_page_sizes)
            .map(|page_size| {
                let page_bytes = 1usize << page_size.0;

                if size % page_bytes == 0 {
                    let pages = size / page_bytes;
                    let actual_size = size;
                    let overhead = 0;
                    (page_size, pages, actual_size, overhead)
                } else {
                    let pages = (size / page_bytes) + 1;
                    let actual_size = page_bytes * pages;
                    let overhead = actual_size - size;
                    (page_size, pages, actual_size, overhead)
                }
            })
            .min_by(|a, b| match a.3.cmp(&b.3) {
                Ordering::Equal => a.1.cmp(&b.1),
                ord => ord,
            });
        let (page_size, pages, actual_size, overhead) = match page_configs {
            None => {
                return Err(QuantumRingError::UnsupportedPageSizes {
                    requested: allowed_page_sizes,
                    supported: supported_page_sizes,
                });
            }
            Some(conf) => conf,
        };
        trace!(
            requested_size = size,
            actual_size,
            overhead,
            pages,
            "requested size: {size}, actual size: {actual_size} using {pages} {:?} pages, overhead: {overhead}",
            PageSizes::from_bits(1usize << page_size.0).unwrap(),
        );

        Self::new(pages, page_size, min_padding)
    }

    pub fn madvise(&self, access_pattern: AccessPattern) -> io::Result<()> {
        unsafe {
            madvise::madvise(
                self.first_map.as_ptr(),
                self.first_map.len()
                    + self.second_map.size()
                    + self
                        .padding_map
                        .as_ref()
                        .map(|p| p.size())
                        .unwrap_or_default(),
                access_pattern,
            )
        }
    }

    #[inline(always)]
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    #[inline(always)]
    pub fn read_len(&self) -> usize {
        self.len
    }

    #[inline(always)]
    pub fn write_len(&self) -> usize {
        self.capacity - self.len
    }

    #[inline(always)]
    pub fn read_slice(&mut self) -> &[u8] {
        unsafe { &*slice_from_raw_parts(self.first_map.as_ptr().add(self.read), self.read_len()) }
    }

    /// # Safety
    /// special care must be taken when advancing the read pointer
    #[inline(always)]
    pub unsafe fn advance_read(&mut self, bytes: usize) {
        debug_assert!(bytes <= self.read_len());

        self.read = (self.read + bytes) % self.capacity;
        self.len -= bytes;

        #[cfg(any(feature = "futures", feature = "tokio"))]
        if let Some(waker) = self.write_waker.take() {
            waker.wake()
        }
    }

    #[inline(always)]
    pub fn write_slice(&mut self) -> &mut [u8] {
        unsafe {
            &mut *slice_from_raw_parts_mut(
                self.first_map.as_mut_ptr().add(self.write),
                self.write_len(),
            )
        }
    }

    /// # Safety
    /// special care must be taken when advancing the write pointer
    #[inline(always)]
    pub unsafe fn advance_write(&mut self, bytes: usize) {
        debug_assert!(bytes <= self.write_len());

        self.write = (self.write + bytes) % self.capacity;
        self.len += bytes;

        #[cfg(any(feature = "futures", feature = "tokio"))]
        if let Some(waker) = self.read_waker.take() {
            waker.wake()
        }
    }
}

impl Read for QuantumRing {
    #[inline(always)]
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        if self.read_len() == 0 {
            return Err(std::io::ErrorKind::Interrupted.into());
        }

        let count = min(self.read_len(), buf.len());
        unsafe {
            std::ptr::copy_nonoverlapping(self.read_slice().as_ptr(), buf.as_mut_ptr(), count);
            self.advance_read(count);
        };
        Ok(count)
    }
}

impl Write for QuantumRing {
    #[inline(always)]
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        if self.write_len() == 0 {
            return Err(std::io::ErrorKind::Interrupted.into());
        }

        let count = min(self.write_len(), buf.len());
        unsafe {
            std::ptr::copy_nonoverlapping(buf.as_ptr(), self.write_slice().as_mut_ptr(), count);
            self.advance_write(count);
        };
        Ok(count)
    }

    #[inline(always)]
    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

#[cfg(feature = "tokio")]
impl tokio_io::AsyncRead for QuantumRing {
    #[inline(always)]
    fn poll_read(
        mut self: std::pin::Pin<&mut Self>,
        cx: &mut Context<'_>,
        buf: &mut [u8],
    ) -> Poll<std::io::Result<usize>> {
        match self.read(buf) {
            Ok(n) => Poll::Ready(Ok(n)),
            Err(e) if e.kind() == std::io::ErrorKind::Interrupted => {
                let _ = self.read_waker.insert(cx.waker().clone());
                Poll::Pending
            }
            Err(e) => Poll::Ready(Err(e)),
        }
    }
}

#[cfg(feature = "futures")]
impl futures_io::AsyncRead for QuantumRing {
    #[inline(always)]
    fn poll_read(
        mut self: std::pin::Pin<&mut Self>,
        cx: &mut Context<'_>,
        buf: &mut [u8],
    ) -> Poll<std::io::Result<usize>> {
        match self.read(buf) {
            Ok(n) => Poll::Ready(Ok(n)),
            Err(e) if e.kind() == std::io::ErrorKind::Interrupted => {
                let _ = self.read_waker.insert(cx.waker().clone());
                Poll::Pending
            }
            Err(e) => Poll::Ready(Err(e)),
        }
    }
}

#[cfg(feature = "tokio")]
impl tokio_io::AsyncWrite for QuantumRing {
    #[inline(always)]
    fn poll_write(
        mut self: std::pin::Pin<&mut Self>,
        cx: &mut Context<'_>,
        buf: &[u8],
    ) -> Poll<std::io::Result<usize>> {
        match self.write(buf) {
            Ok(n) => Poll::Ready(Ok(n)),
            Err(e) if e.kind() == std::io::ErrorKind::Interrupted => {
                let _ = self.write_waker.insert(cx.waker().clone());
                Poll::Pending
            }
            Err(e) => Poll::Ready(Err(e)),
        }
    }

    #[inline(always)]
    fn poll_flush(
        self: std::pin::Pin<&mut Self>,
        _cx: &mut Context<'_>,
    ) -> Poll<std::io::Result<()>> {
        Poll::Ready(Ok(()))
    }

    #[inline(always)]
    fn poll_shutdown(
        self: std::pin::Pin<&mut Self>,
        _cx: &mut Context<'_>,
    ) -> Poll<std::io::Result<()>> {
        Poll::Ready(Ok(()))
    }
}

#[cfg(feature = "futures")]
impl futures_io::AsyncWrite for QuantumRing {
    #[inline(always)]
    fn poll_write(
        mut self: std::pin::Pin<&mut Self>,
        cx: &mut Context<'_>,
        buf: &[u8],
    ) -> Poll<std::io::Result<usize>> {
        match self.write(buf) {
            Ok(n) => Poll::Ready(Ok(n)),
            Err(e) if e.kind() == std::io::ErrorKind::Interrupted => {
                let _ = self.write_waker.insert(cx.waker().clone());
                Poll::Pending
            }
            Err(e) => Poll::Ready(Err(e)),
        }
    }

    #[inline(always)]
    fn poll_flush(
        self: std::pin::Pin<&mut Self>,
        _cx: &mut Context<'_>,
    ) -> Poll<std::io::Result<()>> {
        Poll::Ready(Ok(()))
    }

    #[inline(always)]
    fn poll_close(
        self: std::pin::Pin<&mut Self>,
        _cx: &mut Context<'_>,
    ) -> Poll<std::io::Result<()>> {
        Poll::Ready(Ok(()))
    }
}

#[cfg(test)]
mod test {
    use std::assert_matches::assert_matches;
    use std::io::{ErrorKind, Read, Write};
    use std::iter::zip;
    use std::ops::RangeBounds;

    use mmap_rs::{MmapOptions, PageSizes};
    use rand::RngCore;
    use tracing::Level;

    use crate::{PageSizeIter, QuantumRing};

    #[cfg(test)]
    #[ctor::ctor]
    fn test_global_init() {
        tracing_subscriber::fmt::fmt()
            .compact()
            .without_time()
            .with_max_level(Level::TRACE)
            .init();
    }


    #[test]
    fn does_the_magic_happen() {
        type TestType = usize;
        let test_val: TestType = 0x42069;

        let mut qring =
            QuantumRing::new_with_size(std::mem::size_of_val(&test_val), PageSizes::all(), 0)
                .expect("unable to create QuantumRing");
        assert_ne!(qring.first_map.as_ptr(), qring.second_map.as_ptr());

        unsafe {
            (qring.first_map.as_mut_ptr() as *mut TestType).write(test_val);
            assert_eq!(
                (qring.second_map.as_mut_ptr() as *mut TestType).read(),
                test_val
            );
        }
    }

    #[test]
    fn initial_read_write_len() {
        let qring = QuantumRing::new_with_size(4096, PageSizes::all(), 0)
            .expect("unable to create QuantumRing");
        let size = qring.capacity();

        assert_eq!(qring.read_len(), 0usize);
        assert_eq!(qring.write_len(), size);
    }

    #[test]
    fn read_write() {
        let mut qring = QuantumRing::new_with_size(128, PageSizes::all(), 0)
            .expect("unable to create QuantumRing");
        let size = qring.capacity();
        let test_chunk_size = (size / 2) + (size / 4);
        let mut test_data = vec![0u8; test_chunk_size];
        let mut written = Vec::new();
        let mut read = Vec::new();
        rand::thread_rng().fill_bytes(test_data.as_mut_slice());

        assert_eq!(qring.read_len(), 0usize);
        assert_eq!(qring.write_len(), size);

        assert_matches!(qring.write(test_data.as_slice()), Ok(n) if n == test_chunk_size);
        test_data
            .iter()
            .take(test_chunk_size)
            .for_each(|b| written.push(*b));

        assert_eq!(qring.read_len(), test_chunk_size);
        assert_eq!(qring.write_len(), size - test_chunk_size);

        assert_matches!(qring.write(test_data.as_slice()), Ok(n) if n == size - test_chunk_size);
        test_data
            .iter()
            .take(size - test_chunk_size)
            .for_each(|b| written.push(*b));

        assert_eq!(qring.read_len(), size);
        assert_eq!(qring.write_len(), 0usize);

        assert_matches!(qring.write(test_data.as_slice()), Err(e) if e.kind() == ErrorKind::Interrupted);

        assert_eq!(qring.read_len(), size);
        assert_eq!(qring.write_len(), 0usize);

        let mut part_read = vec![0u8; test_chunk_size];
        assert_matches!(qring.read(part_read.as_mut_slice()), Ok(n) if n == test_chunk_size);
        read.append(&mut part_read);

        assert_eq!(qring.read_len(), size - test_chunk_size);
        assert_eq!(qring.write_len(), test_chunk_size);

        let mut part_read = vec![0u8; test_chunk_size];
        assert_matches!(qring.read(part_read.as_mut_slice()), Ok(n) if n == size - test_chunk_size);
        part_read.truncate(size - test_chunk_size);
        read.append(&mut part_read);

        assert_eq!(qring.read_len(), 0usize);
        assert_eq!(qring.write_len(), size);

        assert_matches!(qring.read(part_read.as_mut_slice()), Err(e) if e.kind() == ErrorKind::Interrupted);

        assert_eq!(qring.read_len(), 0usize);
        assert_eq!(qring.write_len(), size);

        assert_eq!(written.len(), read.len());
        assert!(zip(written.into_iter(), read.into_iter()).all(|(written, read)| written == read));
    }

    #[test]
    fn wrap_around() {
        let mut qring = QuantumRing::new_with_size(128, PageSizes::all(), 42069)
            .expect("unable to create QuantumRing");
        let size = qring.capacity();
        let test_chunk_size = size;
        let mut test_data = vec![0u8; test_chunk_size];
        let mut written = Vec::new();
        let mut read = Vec::new();
        rand::thread_rng().fill_bytes(test_data.as_mut_slice());

        assert_eq!(qring.read_len(), 0usize);
        assert_eq!(qring.write_len(), size);

        assert_matches!(qring.write(&test_data.as_slice()[0..(test_chunk_size / 2)]), Ok(n) if n == test_chunk_size / 2);
        test_data
            .iter()
            .take(test_chunk_size / 2)
            .for_each(|b| written.push(*b));

        assert_eq!(qring.read_len(), test_chunk_size / 2);
        assert_eq!(qring.write_len(), size - (test_chunk_size / 2));

        let mut part_read = vec![0u8; test_chunk_size];
        assert_matches!(qring.read(part_read.as_mut_slice()), Ok(n) if n == test_chunk_size / 2);
        part_read.truncate(test_chunk_size / 2);
        read.append(&mut part_read);

        assert_eq!(qring.read_len(), 0);
        assert_eq!(qring.write_len(), size);

        // this write includes a wrap around
        assert_matches!(qring.write(&test_data.as_slice()[0..test_chunk_size]), Ok(n) if n == test_chunk_size);
        test_data
            .iter()
            .take(test_chunk_size)
            .for_each(|b| written.push(*b));

        assert_eq!(qring.read_len(), test_chunk_size);
        assert_eq!(qring.write_len(), 0usize);

        let mut part_read = vec![0u8; test_chunk_size];
        assert_matches!(qring.read(part_read.as_mut_slice()), Ok(n) if n == test_chunk_size);
        read.append(&mut part_read);

        assert_eq!(qring.read_len(), 0);
        assert_eq!(qring.write_len(), size);

        assert_eq!(written.len(), read.len());
        assert!(zip(written.into_iter(), read.into_iter()).all(|(written, read)| written == read));
    }

    #[test]
    fn read_with_lookahead_does_not_segfault() {
        let mut qring = QuantumRing::new(
            1,
            PageSizeIter::new(
                MmapOptions::page_sizes().expect("unable to get supported page sizes"),
            )
            .next()
            .expect("no page sizes available"),
            std::mem::size_of::<usize>(),
        )
        .expect("unable to create QuantumRing");

        assert_matches!(qring.padding_map, Some(ref map) if map.size() > 0);

        unsafe {
            qring.advance_write(qring.write_len() - 1);
            qring.advance_read(qring.read_len());
            qring.advance_write(qring.write_len());

            assert_eq!(qring.write_len(), 0);
            assert_eq!(qring.read_len(), qring.capacity());

            let read_slice_end_ptr = qring
                .read_slice()
                .as_ptr()
                .add(qring.read_len() - 1);

            // suppose we are at the last byte of the read_slice (read_slice_end_ptr) and we need
            // to read a usize but we do not want to crash.
            // So we should point into our padding.
            assert_matches!(qring.padding_map, Some(ref map) if map.as_ptr_range().contains(&read_slice_end_ptr.add(std::mem::size_of::<usize>())));

            _ = std::hint::black_box((read_slice_end_ptr as *const usize).read_unaligned());

            // we still alive? no segfault??
            assert!(true);
        }
    }
}
