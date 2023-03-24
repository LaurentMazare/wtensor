use clap::Parser;

#[derive(Parser)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// The matrix size to be used.
    #[arg(long, default_value = "1024")]
    size: usize,
}

async fn run(args: Args) -> anyhow::Result<()> {
    let result = execute_gpu(args).await?;
    println!("Mm result: [{:?}]", &result[..result.len().min(10)]);
    Ok(())
}

async fn execute_gpu(args: Args) -> anyhow::Result<Vec<f32>> {
    let device = wtensor::Device::new().await?;
    execute_gpu_inner(&device, args).await
}

async fn execute_gpu_inner(d: &wtensor::Device, args: Args) -> anyhow::Result<Vec<f32>> {
    let sz = args.size;
    let m1 = wtensor::Tensor2D::<f32>::new(d, sz, sz, 1.1);
    let m2 = wtensor::Tensor2D::<f32>::new(d, sz, sz, 2.);
    let start = std::time::Instant::now();
    let m1m2 = m1.matmul(&m2)?;
    let result = m1m2.to_vec().await?;
    let duration = start.elapsed();
    let gflops = (sz as f32).powi(3) / duration.as_secs_f32() * 2e-9;
    println!("Ran computation in {duration:?} {gflops:.2} GFLOPS.");
    let m1m2 = m1m2.add(&m1m2)?.to_vec().await?;
    println!("Mm*2 result: [{:?}]", &m1m2[..m1m2.len().min(10)]);
    Ok(result)
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    pollster::block_on(run(args))?;
    Ok(())
}
