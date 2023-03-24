/// Main library error type.
#[derive(thiserror::Error, Debug)]
pub enum Error {
    /// I/O error.
    #[error(transparent)]
    Io(#[from] std::io::Error),

    /// WebGPU request device errors.
    #[error(transparent)]
    RequestDeviceError(#[from] wgpu::RequestDeviceError),

    /// WebGPU buffer async errors.
    #[error(transparent)]
    BufferAsyncError(#[from] wgpu::BufferAsyncError),

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
}

pub type Result<T> = std::result::Result<T, Error>;
