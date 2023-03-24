@group(0)
@binding(0)
var<storage,read_write> fill_buffer: array<f32>;

@group(0)
@binding(1)
var<storage,read> fill_value: array<f32>;

@compute
@workgroup_size(16, 16, 1)
fn fill(@builtin(global_invocation_id) global_id: vec3<u32>) {
  var x: u32 = global_id.x * 1024u + global_id.y;
  let len: u32 = arrayLength(&fill_buffer);
  let start_k: u32 = x * 64u;
  let end_k: u32 = min(start_k + 64u, len);
  let value = fill_value[0];
  for(var k: u32 = start_k; k < end_k; k += 1u) {
    fill_buffer[k] = value;
  }
}
