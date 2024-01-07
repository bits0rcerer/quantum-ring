<h1 align="center">💠 Quantum Ring 💠</h1>
<p align="center">
<img style="margin: auto" src="./logo.jpeg" width="256">
</p>
<p align="center">An easy-to-use low-level ring buffer implementation that leverages virtual address space trickery for Rust 🦀</p>

## How to

- Create a new `QuantumRing` with a `size`
    - `size` will be treated as a lower boundary. The actual size may be greater, because the buffer has to be page
      aligned.
  ```rust
  let qring = QuantumRing::new_with_size(1024, PageSizes::all())
                .expect("unable to create QuantumRing");
  ```
- Read/Write
    - Read using direct access
  ```rust
  let slice: &[u8] = qring.read_slice();
  // do something with 0 <= n < slice.len() == qring.read_len() bytes
  unsafe { qring.advance_read(n) };
  ```  
    - Write using direct access
  ```rust
  let slice: &mut [u8] = qring.write_slice();
  // overwrite 0 <= n < slice.len() == qring.write_len() bytes from slice
  unsafe { qring.advance_write(n) };
  ```
    - `QuantumRing` implements `std::io::Read`
    - `QuantumRing` implements `std::io::Write`
    - `QuantumRing` implements `futures_io::AsyncRead` (feature: `futures`)
    - `QuantumRing` implements `futures_io::AsyncWrite` (feature: `futures`)
    - `QuantumRing` implements `tokio_io::AsyncRead` (feature: `tokio`)
    - `QuantumRing` implements `tokio_io::AsyncWrite` (feature: `tokio`)
