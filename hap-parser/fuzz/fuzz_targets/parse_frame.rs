#![no_main]

//! Fuzz the HAP frame parser against arbitrary bytes.
//!
//! The parser takes untrusted input (one packet out of a container), so the
//! only invariant we assert is *it never panics* — every input must return
//! `Ok` or a `HapError`, never unwind. Run with:
//!
//! ```text
//! cargo +nightly fuzz run parse_frame
//! ```

use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    let _ = hap_parser::parse_frame(data);
    let _ = hap_parser::detect_format(data);
});
