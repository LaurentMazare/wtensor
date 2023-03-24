struct GemmDims {
  M: u32,
  K: u32,
  N: u32,
}

@group(0)
@binding(0)
var<storage,read> gemm_lhs: array<f32>;

@group(0)
@binding(1)
var<storage,read> gemm_rhs: array<f32>;

@group(0)
@binding(2)
var<storage,read_write> gemm_output: array<f32>;

@group(0)
@binding(3)
var<storage,read> gemm_dims: GemmDims;

@compute
@workgroup_size(16, 16, 1)
fn gemm(@builtin(global_invocation_id) global_id: vec3<u32>) {
  var x: u32 = global_id.x;
  var y: u32 = global_id.y;

  var M: u32 = gemm_dims.M;
  var K: u32 = gemm_dims.K;
  var N: u32 = gemm_dims.N;

  if (x >= N || y >= M) {
    return;
  }

  let y_K: u32 = y * K;
  var k_N: u32 = 0u;
  var sum: f32 = 0.0;
  for(var k: u32 = 0u; k < K; k += 1u) {
    sum += gemm_lhs[y_K + k] * gemm_rhs[k_N + x];
    k_N += N;
  }
  gemm_output[x + y * N] = sum;
}
