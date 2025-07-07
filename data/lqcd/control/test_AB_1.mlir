func.func private @nanoTime() -> i64 attributes { llvm.emit_c_interface }
func.func @main(%B_28: memref<1024x1024xf64>, %A_30: memref<1024x1024xf64>, %output_24: memref<1024x1024xf64>) -> i64 attributes { llvm.emit_c_interface } {
  %t0 = func.call @nanoTime() : () -> i64
  %7 = memref.alloc() : memref<1024x1024x1xf64>
  linalg.generic {indexing_maps = [affine_map<(o_0_20, o_1_22, complex_26)->(o_0_20, o_1_22, complex_26)>], iterator_types = ["parallel", "parallel", "parallel"]} outs(%7: memref<1024x1024x1xf64>) {
    ^bb0(%8: f64):
    %1 = arith.constant 0.0 : f64
    linalg.yield %1 : f64
  }
  %9 = memref.alloc() : memref<1024xf64>
  linalg.generic {indexing_maps = [affine_map<(o_0_20, o_1_22, complex_26, sum_36_0)->(sum_36_0)>, affine_map<(o_0_20, o_1_22, complex_26, sum_36_0)->(o_0_20, sum_36_0)>, affine_map<(o_0_20, o_1_22, complex_26, sum_36_0)->(sum_36_0, o_1_22)>, affine_map<(o_0_20, o_1_22, complex_26, sum_36_0)->(o_0_20, sum_36_0)>, affine_map<(o_0_20, o_1_22, complex_26, sum_36_0)->(sum_36_0, o_1_22)>, affine_map<(o_0_20, o_1_22, complex_26, sum_36_0)->(o_0_20, o_1_22, complex_26)>], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%9, %A_30, %B_28, %A_30, %B_28: memref<1024xf64>, memref<1024x1024xf64>, memref<1024x1024xf64>, memref<1024x1024xf64>, memref<1024x1024xf64>) outs(%7: memref<1024x1024x1xf64>) {
    ^bb0(%10: f64, %13: f64, %37: f64, %30: f64, %39: f64, %11: f64):
    %4 = linalg.index 0 : index
    %5 = linalg.index 1 : index
    %6 = linalg.index 2 : index
    %12 = linalg.index 3 : index
    %27 = arith.constant 1 : index
    %32 = arith.minsi %6, %27 : index
    %17 = arith.constant 0 : index
    %33 = arith.maxsi %32, %17 : index
    %26 = arith.mulf %13, %37 fastmath<nnan, ninf, nsz, reassoc, contract, afn> : f64
    %22 = arith.constant 0.0 : f64
    %18 = arith.subf %26, %22 fastmath<nnan, ninf, nsz, reassoc, contract, afn> : f64
    %29 = arith.constant 0.0 : f64
    %24 = arith.mulf %30, %29 fastmath<nnan, ninf, nsz, reassoc, contract, afn> : f64
    %15 = arith.constant 0.0 : f64
    %14 = arith.mulf %15, %39 fastmath<nnan, ninf, nsz, reassoc, contract, afn> : f64
    %19 = arith.addf %24, %14 fastmath<nnan, ninf, nsz, reassoc, contract, afn> : f64
    %36 = arith.constant 0 : index
    %21 = arith.cmpi eq, %33, %36 : index
    %38 = arith.select %21, %18, %19 : f64
    %25 = arith.addf %11, %38 fastmath<nnan, ninf, nsz, reassoc, contract, afn> : f64
    linalg.yield %25 : f64
  }
  %41 = memref.collapse_shape %7 [[0], [1, 2]] : memref<1024x1024x1xf64> into memref<1024x1024xf64>
  linalg.generic {indexing_maps = [affine_map<(o_0_20, o_1_22)->(o_0_20, o_1_22)>, affine_map<(o_0_20, o_1_22)->(o_0_20, o_1_22)>], iterator_types = ["parallel", "parallel"]} ins(%41: memref<1024x1024xf64>) outs(%output_24: memref<1024x1024xf64>) {
    ^bb0(%43: f64, %42: f64):
    %2 = linalg.index 0 : index
    %3 = linalg.index 1 : index
    linalg.yield %43 : f64
  }
  %t1 = func.call @nanoTime() : () -> (i64)
  %t2 = arith.subi %t1, %t0 : i64
  return %t2 : i64
}
