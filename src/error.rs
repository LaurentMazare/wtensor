/// Main library error type.
#[derive(thiserror::Error, Debug)]
pub enum Error {
    /// I/O error.
    #[error(transparent)]
    Io(#[from] std::io::Error),

    /// WebGPU request device errors.
    #[error(transparent)]
    RequestDevice(#[from] wgpu::RequestDeviceError),

    /// WebGPU buffer async errors.
    #[error(transparent)]
    BufferAsync(#[from] wgpu::BufferAsyncError),

    /// Receiver returned None.
    #[error("receiver returned None")]
    ReceiverReturnedNone,

    /// No adapter available.
    #[error("no adapter for instance {0:?}")]
    NoAdapter(wgpu::Instance),

    /// Dimension mismatch.
    #[error("dimension mismatch in binary operator {op}: {lhs:?} {rhs:?}")]
    DimensionMismatchBinaryOp {
        op: &'static str,
        lhs: Vec<usize>,
        rhs: Vec<usize>,
    },

    /// Internal error, these only happen when encountering a bug within this crate.
    #[error("internal error {0:?}")]
    InternalError(String),
}

pub type Result<T> = std::result::Result<T, Error>;

macro_rules! ierr {
    ($($args: tt)*) => {
        Error::InternalError(format!($($args)*))
    }
}

pub(crate) use ierr;
