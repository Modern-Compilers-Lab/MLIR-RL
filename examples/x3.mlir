// RUN: mlir-opt %s --test-transform-dialect-interpreter \
// RUN:             --test-transform-dialect-erase-schedule \
// RUN:             --math-uplift-to-fma \
// RUN:             --convert-bufferization-to-memref \
// RUN:             --test-lower-to-llvm |\
// RUN: FileCheck %s

// Fixed-size tensor types to be used in convolution.
// Named sizes are: N=5 OH=80 OW=100 F=C=128 KH=KW=3.
// Input is NHWC.
// Filter is CHWF.
// Ouptut is NHWF.
!tinput = tensor<5x82x102x128xf32>
!tfilter = tensor<128x3x3x128xf32>
!tbias = tensor<128xf32>
!toutput = tensor<5x80x100x128xf32>

// Function containing the convolution. Note that its arguments and results are
// tensors annotated with attributes from the `bufferization` dialect. These
// attributes hint the bufferization pass to assume buffers can be directly
// used for these tensors without reshaping.
func.func @conv(
    %input: !tinput {bufferization.writable = false,
                     bufferization.access = "read",
                     bufferization.buffer_layout =
                         affine_map<(d0,d1,d2,d3)->(d0,d1,d2,d3)>},
    %filter: !tfilter {bufferization.writable = false,
                      bufferization.access = "read",
                      bufferization.buffer_layout =
                          affine_map<(d0,d1,d2,d3)->(d0,d1,d2,d3)>},
    %bias: !tbias {bufferization.writable = false,
                   bufferization.access = "read",
                   bufferization.buffer_layout = affine_map<(d0)->(d0)>},
    %output: !toutput {bufferization.writable = true,
                       bufferization.buffer_layout =
                           affine_map<(d0,d1,d2,d3)->(d0,d1,d2,d3)>,
                       bufferization.access = "write"}) -> !toutput
  // This requests a C-compatible interface to be emitted for the function
  // when translating to LLVM IR.
  attributes { llvm.emit_c_interface }
{
  // Bias. Using a named Linalg operation for brevity.
  %bias_init = tensor.empty() : !toutput
  %biased = linalg.broadcast ins(%bias : !tbias)
    outs(%bias_init : !toutput) dimensions = [0, 1, 2]

  // Convolution proper. While Linalg has named operations for 2D convolutions,
  // the one in the Halide example has an uncommon order of filter dimensions
  // and is not supported. It also takes the fitler as first argument. This
  // code recreates it faithfully using the generic form.
  %convolved = linalg.generic {
    iterator_types = ["parallel", "parallel", "parallel", "parallel",
                      "reduction", "reduction", "reduction"],
    indexing_maps = [
      affine_map<(n, y, x, c, rz, ry, rx) -> (rx, rz, ry, c)>,
      affine_map<(n, y, x, c, rz, ry, rx) -> (n, y+rz, x+ry, rx)>,
      affine_map<(n, y, x, c, rz, ry, rx) -> (n, y, x, c)>
    ]
  } ins(%filter, %input: !tfilter, !tinput) outs(%biased : !toutput) {
  ^bb0(%in: f32, %f: f32, %b: f32):
    // Note the fastmath attributes that allow operations to be recombined into
    //   %0 = math.fma %in, %f, %b : f32
    // later on and to reorder reductions.
    %m1 = arith.mulf %in, %f  {fastmath = #arith.fastmath<fast>} : f32
    %0 = arith.addf %b, %m1  {fastmath = #arith.fastmath<fast>} : f32
    linalg.yield %0 : f32
  } -> !toutput

  // ReLU is just a max(0, x).
  %c0 = arith.constant 0.0 : f32
  %relued = linalg.generic {
    iterator_types = ["parallel", "parallel", "parallel", "parallel"],
    indexing_maps = [
      affine_map<(d0, d1, d2, d3) -> ()>,
      affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
      affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
    ]
  } ins(%c0, %convolved : f32, !toutput)
    outs(%output : !toutput) {
  ^bb0(%cst: f32, %in: f32, %out: f32):
    %0 = llvm.intr.maxnum(%cst, %in) : (f32, f32) -> f32
    linalg.yield %0 : f32
  } -> !toutput

  return %relued : !toutput
}



module attributes { transform.with_named_sequence } {

  transform.sequence failures(propagate) {
  // This argument will point to the top-level module.
  ^bb0(%arg0: !transform.any_op):
    %bias = transform.structured.match ops{["linalg.broadcast"]} in %arg0
      : (!transform.any_op) -> !transform.any_op
    %generics = transform.structured.match ops{["linalg.generic"]} in %arg0
      : (!transform.any_op) -> !transform.any_op
    %conv, %relu = transform.split_handle %generics
      : (!transform.any_op) -> (!transform.any_op, !transform.any_op)

    %relu2, %co     = transform.structured.tile_using_forall %relu  tile_sizes [0, 0, 0, 64] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    %relu3, %n_y_xo = transform.structured.tile_using_forall %relu2 tile_sizes [1, 1, 5,  0] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)

    %conv2, %co2 = transform.structured.fuse_into_containing_op %conv into %co
      : (!transform.any_op, !transform.any_op)
      -> (!transform.any_op, !transform.any_op)
    %conv3, %n_y_xo2 = transform.structured.fuse_into_containing_op %conv2
      into %n_y_xo
      : (!transform.any_op, !transform.any_op)
      -> (!transform.any_op, !transform.any_op)

    // Also fuse the bias that we represent as a separate operation and Halide
    // represents as the "pure" (as opposed to "update") part of the conv
    // expression. Note that fusion consumes both handles and produces new
    // handles for chaining purposes.
    %bias2, %co3 = transform.structured.fuse_into_containing_op %bias into %co2
      : (!transform.any_op, !transform.any_op)
      -> (!transform.any_op, !transform.any_op)
    %bias3, %n_y_xo3 = transform.structured.fuse_into_containing_op %bias2 into %n_y_xo2
      : (!transform.any_op, !transform.any_op)
      -> (!transform.any_op, !transform.any_op)


  }
}
