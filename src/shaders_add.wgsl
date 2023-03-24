@group(0)
@binding(0)
var<storage,read> add_lhs: array<f32>;

@group(0)
@binding(1)
var<storage,read> add_rhs: array<f32>;

@group(0)
@binding(2)
var<storage,read_write> add_output: array<f32>;

@compute
@workgroup_size(256, 1, 1)
fn add(@builtin(global_invocation_id) global_id: vec3<u32>) {
  var x: u32 = global_id.x;
  let len: u32 = arrayLength(&add_lhs);
  let start_k: u32 = x * 64u;
  let end_k: u32 = min(start_k + 64u, len);
  for(var k: u32 = start_k; k < end_k; k += 1u) {
    add_output[k] = add_lhs[k] + add_rhs[k];
  }
}
