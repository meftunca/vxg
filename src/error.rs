use std::fmt;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Error {
    InvalidOption(&'static str),
    InvalidFrame(&'static str),
    Truncated,
    SizeOverflow,
    OutputLimit { requested: u64, limit: usize },
    OutputSize { expected: usize, actual: usize },
    Checksum { block: usize },
    Unsupported(String),
    Backend(String),
    WorkerPanic,
}
impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidOption(s) => write!(f, "invalid option: {s}"),
            Self::InvalidFrame(s) => write!(f, "invalid VXG frame: {s}"),
            Self::Truncated => f.write_str("truncated VXG frame"),
            Self::SizeOverflow => f.write_str("integer or allocation-size overflow"),
            Self::OutputLimit { requested, limit } => write!(f, "output {requested} exceeds limit {limit}"),
            Self::OutputSize { expected, actual } => write!(f, "output length {actual}, expected {expected}"),
            Self::Checksum { block } => write!(f, "CRC32C mismatch in block {block}"),
            Self::Unsupported(s) => write!(f, "unsupported: {s}"),
            Self::Backend(s) => write!(f, "GPU backend: {s}"),
            Self::WorkerPanic => f.write_str("codec worker panicked"),
        }
    }
}
impl std::error::Error for Error {}
pub type Result<T> = std::result::Result<T, Error>;
