module attributes {torch.debug_module_name = "Net"} {
  llvm.func @free(!llvm.ptr)
  llvm.func @memrefCopy(i64, !llvm.ptr, !llvm.ptr)
  llvm.func @malloc(i64) -> !llvm.ptr
  llvm.func private @nanoTime() -> i64 attributes {llvm.emit_c_interface, sym_visibility = "private"} {
    %0 = llvm.call @_mlir_ciface_nanoTime() : () -> i64
    llvm.return %0 : i64
  }
  llvm.func @_mlir_ciface_nanoTime() -> i64 attributes {llvm.emit_c_interface, sym_visibility = "private"}
  llvm.func @printFlops(f64) attributes {sym_visibility = "private"}
  llvm.func @printI64(i64) attributes {sym_visibility = "private"}
  llvm.func @printNewline() attributes {sym_visibility = "private"}
  llvm.func @printMemrefF32(i64, !llvm.ptr) attributes {sym_visibility = "private"}
  llvm.func @matmul() -> !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> {
    %0 = llvm.mlir.constant(4 : i64) : i64
    %1 = llvm.mlir.constant(16 : index) : i64
    %2 = llvm.mlir.constant(802816 : index) : i64
    %3 = llvm.mlir.constant(12544 : index) : i64
    %4 = llvm.mlir.constant(147 : index) : i64
    %5 = llvm.mlir.constant(49 : index) : i64
    %6 = llvm.mlir.constant(158700 : index) : i64
    %7 = llvm.mlir.constant(52900 : index) : i64
    %8 = llvm.mlir.constant(2.000000e+00 : f32) : f32
    %9 = llvm.mlir.constant(0 : index) : i64
    %10 = llvm.mlir.constant(32 : index) : i64
    %11 = llvm.mlir.constant(1 : index) : i64
    %12 = llvm.mlir.constant(3 : index) : i64
    %13 = llvm.mlir.constant(230 : index) : i64
    %14 = llvm.mlir.constant(64 : index) : i64
    %15 = llvm.mlir.constant(7 : index) : i64
    %16 = llvm.mlir.constant(112 : index) : i64
    %17 = llvm.mlir.constant(2 : index) : i64
    %18 = llvm.mlir.constant(8 : index) : i64
    %19 = llvm.mlir.constant(28 : index) : i64
    %20 = llvm.mlir.constant(14 : index) : i64
    %21 = llvm.mlir.zero : !llvm.ptr
    %22 = llvm.getelementptr %21[5078400] : (!llvm.ptr) -> !llvm.ptr, f32
    %23 = llvm.ptrtoint %22 : !llvm.ptr to i64
    %24 = llvm.add %23, %14  : i64
    %25 = llvm.call @malloc(%24) : (i64) -> !llvm.ptr
    %26 = llvm.ptrtoint %25 : !llvm.ptr to i64
    %27 = llvm.sub %14, %11  : i64
    %28 = llvm.add %26, %27  : i64
    %29 = llvm.urem %28, %14  : i64
    %30 = llvm.sub %28, %29  : i64
    %31 = llvm.inttoptr %30 : i64 to !llvm.ptr
    omp.parallel {
      omp.wsloop for  (%arg0, %arg1, %arg2, %arg3) : i64 = (%9, %9, %9, %9) to (%10, %12, %13, %13) step (%11, %11, %11, %11) {
        %142 = llvm.mul %arg0, %6  : i64
        %143 = llvm.mul %arg1, %7  : i64
        %144 = llvm.add %142, %143  : i64
        %145 = llvm.mul %arg2, %13  : i64
        %146 = llvm.add %144, %145  : i64
        %147 = llvm.add %146, %arg3  : i64
        %148 = llvm.getelementptr %31[%147] : (!llvm.ptr, i64) -> !llvm.ptr, f32
        llvm.store %8, %148 : f32, !llvm.ptr
        omp.yield
      }
      omp.terminator
    }
    %32 = llvm.mlir.zero : !llvm.ptr
    %33 = llvm.getelementptr %32[9408] : (!llvm.ptr) -> !llvm.ptr, f32
    %34 = llvm.ptrtoint %33 : !llvm.ptr to i64
    %35 = llvm.add %34, %14  : i64
    %36 = llvm.call @malloc(%35) : (i64) -> !llvm.ptr
    %37 = llvm.ptrtoint %36 : !llvm.ptr to i64
    %38 = llvm.sub %14, %11  : i64
    %39 = llvm.add %37, %38  : i64
    %40 = llvm.urem %39, %14  : i64
    %41 = llvm.sub %39, %40  : i64
    %42 = llvm.inttoptr %41 : i64 to !llvm.ptr
    omp.parallel {
      omp.wsloop for  (%arg0, %arg1, %arg2, %arg3) : i64 = (%9, %9, %9, %9) to (%14, %12, %15, %15) step (%11, %11, %11, %11) {
        %142 = llvm.mul %arg0, %4  : i64
        %143 = llvm.mul %arg1, %5  : i64
        %144 = llvm.add %142, %143  : i64
        %145 = llvm.mul %arg2, %15  : i64
        %146 = llvm.add %144, %145  : i64
        %147 = llvm.add %146, %arg3  : i64
        %148 = llvm.getelementptr %42[%147] : (!llvm.ptr, i64) -> !llvm.ptr, f32
        llvm.store %8, %148 : f32, !llvm.ptr
        omp.yield
      }
      omp.terminator
    }
    %43 = llvm.mlir.zero : !llvm.ptr
    %44 = llvm.getelementptr %43[25690112] : (!llvm.ptr) -> !llvm.ptr, f32
    %45 = llvm.ptrtoint %44 : !llvm.ptr to i64
    %46 = llvm.add %45, %14  : i64
    %47 = llvm.call @malloc(%46) : (i64) -> !llvm.ptr
    %48 = llvm.ptrtoint %47 : !llvm.ptr to i64
    %49 = llvm.sub %14, %11  : i64
    %50 = llvm.add %48, %49  : i64
    %51 = llvm.urem %50, %14  : i64
    %52 = llvm.sub %50, %51  : i64
    %53 = llvm.inttoptr %52 : i64 to !llvm.ptr
    %54 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
    %55 = llvm.insertvalue %47, %54[0] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %56 = llvm.insertvalue %53, %55[1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %57 = llvm.insertvalue %9, %56[2] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %58 = llvm.insertvalue %10, %57[3, 0] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %59 = llvm.insertvalue %14, %58[3, 1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %60 = llvm.insertvalue %16, %59[3, 2] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %61 = llvm.insertvalue %16, %60[3, 3] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %62 = llvm.insertvalue %2, %61[4, 0] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %63 = llvm.insertvalue %3, %62[4, 1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %64 = llvm.insertvalue %16, %63[4, 2] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %65 = llvm.insertvalue %11, %64[4, 3] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    omp.parallel {
      omp.wsloop for  (%arg0, %arg1, %arg2, %arg3) : i64 = (%9, %9, %9, %9) to (%10, %14, %16, %16) step (%11, %11, %11, %11) {
        %142 = llvm.mul %arg0, %2  : i64
        %143 = llvm.mul %arg1, %3  : i64
        %144 = llvm.add %142, %143  : i64
        %145 = llvm.mul %arg2, %16  : i64
        %146 = llvm.add %144, %145  : i64
        %147 = llvm.add %146, %arg3  : i64
        %148 = llvm.getelementptr %53[%147] : (!llvm.ptr, i64) -> !llvm.ptr, f32
        llvm.store %8, %148 : f32, !llvm.ptr
        omp.yield
      }
      omp.terminator
    }
    %66 = llvm.call @nanoTime() : () -> i64
    llvm.br ^bb1(%9 : i64)
  ^bb1(%67: i64):  // 2 preds: ^bb0, ^bb11
    %68 = llvm.icmp "slt" %67, %1 : i64
    llvm.cond_br %68, ^bb2, ^bb12
  ^bb2:  // pred: ^bb1
    llvm.br ^bb3(%9 : i64)
  ^bb3(%69: i64):  // 2 preds: ^bb2, ^bb10
    %70 = llvm.icmp "slt" %69, %18 : i64
    llvm.cond_br %70, ^bb4, ^bb11
  ^bb4:  // pred: ^bb3
    llvm.br ^bb5(%9 : i64)
  ^bb5(%71: i64):  // 2 preds: ^bb4, ^bb9
    %72 = llvm.icmp "slt" %71, %18 : i64
    llvm.cond_br %72, ^bb6, ^bb10
  ^bb6:  // pred: ^bb5
    llvm.br ^bb7(%9 : i64)
  ^bb7(%73: i64):  // 2 preds: ^bb6, ^bb8
    %74 = llvm.icmp "slt" %73, %18 : i64
    llvm.cond_br %74, ^bb8, ^bb9
  ^bb8:  // pred: ^bb7
    %75 = llvm.mul %67, %17  : i64
    %76 = llvm.mul %71, %19  : i64
    %77 = llvm.mul %73, %19  : i64
    %78 = llvm.mul %69, %18  : i64
    %79 = llvm.mul %71, %20  : i64
    %80 = llvm.mul %73, %20  : i64
    %81 = llvm.mul %75, %6  : i64
    %82 = llvm.mul %76, %13  : i64
    %83 = llvm.add %81, %82  : i64
    %84 = llvm.add %83, %77  : i64
    %85 = llvm.mul %78, %4  : i64
    %86 = llvm.mul %75, %2  : i64
    %87 = llvm.mul %78, %3  : i64
    %88 = llvm.add %86, %87  : i64
    %89 = llvm.mul %79, %16  : i64
    %90 = llvm.add %88, %89  : i64
    %91 = llvm.add %90, %80  : i64
    %92 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
    %93 = llvm.insertvalue %47, %92[0] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %94 = llvm.insertvalue %53, %93[1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %95 = llvm.insertvalue %91, %94[2] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %96 = llvm.insertvalue %17, %95[3, 0] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %97 = llvm.insertvalue %2, %96[4, 0] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %98 = llvm.insertvalue %18, %97[3, 1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %99 = llvm.insertvalue %3, %98[4, 1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %100 = llvm.insertvalue %20, %99[3, 2] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %101 = llvm.insertvalue %16, %100[4, 2] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %102 = llvm.insertvalue %20, %101[3, 3] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %103 = llvm.insertvalue %11, %102[4, 3] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    omp.parallel {
      omp.wsloop for  (%arg0, %arg1, %arg2, %arg3) : i64 = (%9, %9, %9, %9) to (%17, %18, %20, %20) step (%11, %11, %11, %11) {
        llvm.br ^bb1(%9 : i64)
      ^bb1(%142: i64):  // 2 preds: ^bb0, ^bb8
        %143 = llvm.icmp "slt" %142, %12 : i64
        llvm.cond_br %143, ^bb2, ^bb9
      ^bb2:  // pred: ^bb1
        llvm.br ^bb3(%9 : i64)
      ^bb3(%144: i64):  // 2 preds: ^bb2, ^bb7
        %145 = llvm.icmp "slt" %144, %15 : i64
        llvm.cond_br %145, ^bb4, ^bb8
      ^bb4:  // pred: ^bb3
        llvm.br ^bb5(%9 : i64)
      ^bb5(%146: i64):  // 2 preds: ^bb4, ^bb6
        %147 = llvm.icmp "slt" %146, %15 : i64
        llvm.cond_br %147, ^bb6, ^bb7
      ^bb6:  // pred: ^bb5
        %148 = llvm.mul %arg2, %17  : i64
        %149 = llvm.add %148, %144  : i64
        %150 = llvm.mul %arg3, %17  : i64
        %151 = llvm.add %150, %146  : i64
        %152 = llvm.getelementptr %31[%84] : (!llvm.ptr, i64) -> !llvm.ptr, f32
        %153 = llvm.mul %arg0, %6  : i64
        %154 = llvm.mul %142, %7  : i64
        %155 = llvm.add %153, %154  : i64
        %156 = llvm.mul %149, %13  : i64
        %157 = llvm.add %155, %156  : i64
        %158 = llvm.add %157, %151  : i64
        %159 = llvm.getelementptr %152[%158] : (!llvm.ptr, i64) -> !llvm.ptr, f32
        %160 = llvm.load %159 : !llvm.ptr -> f32
        %161 = llvm.getelementptr %42[%85] : (!llvm.ptr, i64) -> !llvm.ptr, f32
        %162 = llvm.mul %arg1, %4  : i64
        %163 = llvm.mul %142, %5  : i64
        %164 = llvm.add %162, %163  : i64
        %165 = llvm.mul %144, %15  : i64
        %166 = llvm.add %164, %165  : i64
        %167 = llvm.add %166, %146  : i64
        %168 = llvm.getelementptr %161[%167] : (!llvm.ptr, i64) -> !llvm.ptr, f32
        %169 = llvm.load %168 : !llvm.ptr -> f32
        %170 = llvm.getelementptr %53[%91] : (!llvm.ptr, i64) -> !llvm.ptr, f32
        %171 = llvm.mul %arg0, %2  : i64
        %172 = llvm.mul %arg1, %3  : i64
        %173 = llvm.add %171, %172  : i64
        %174 = llvm.mul %arg2, %16  : i64
        %175 = llvm.add %173, %174  : i64
        %176 = llvm.add %175, %arg3  : i64
        %177 = llvm.getelementptr %170[%176] : (!llvm.ptr, i64) -> !llvm.ptr, f32
        %178 = llvm.load %177 : !llvm.ptr -> f32
        %179 = llvm.fmul %160, %169  : f32
        %180 = llvm.fadd %178, %179  : f32
        %181 = llvm.getelementptr %53[%91] : (!llvm.ptr, i64) -> !llvm.ptr, f32
        %182 = llvm.mul %arg0, %2  : i64
        %183 = llvm.mul %arg1, %3  : i64
        %184 = llvm.add %182, %183  : i64
        %185 = llvm.mul %arg2, %16  : i64
        %186 = llvm.add %184, %185  : i64
        %187 = llvm.add %186, %arg3  : i64
        %188 = llvm.getelementptr %181[%187] : (!llvm.ptr, i64) -> !llvm.ptr, f32
        llvm.store %180, %188 : f32, !llvm.ptr
        %189 = llvm.add %146, %11  : i64
        llvm.br ^bb5(%189 : i64)
      ^bb7:  // pred: ^bb5
        %190 = llvm.add %144, %11  : i64
        llvm.br ^bb3(%190 : i64)
      ^bb8:  // pred: ^bb3
        %191 = llvm.add %142, %11  : i64
        llvm.br ^bb1(%191 : i64)
      ^bb9:  // pred: ^bb1
        omp.yield
      }
      omp.terminator
    }
    %104 = llvm.mul %75, %2  : i64
    %105 = llvm.mul %78, %3  : i64
    %106 = llvm.add %104, %105  : i64
    %107 = llvm.mul %79, %16  : i64
    %108 = llvm.add %106, %107  : i64
    %109 = llvm.add %108, %80  : i64
    %110 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
    %111 = llvm.insertvalue %47, %110[0] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %112 = llvm.insertvalue %53, %111[1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %113 = llvm.insertvalue %109, %112[2] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %114 = llvm.insertvalue %17, %113[3, 0] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %115 = llvm.insertvalue %2, %114[4, 0] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %116 = llvm.insertvalue %18, %115[3, 1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %117 = llvm.insertvalue %3, %116[4, 1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %118 = llvm.insertvalue %20, %117[3, 2] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %119 = llvm.insertvalue %16, %118[4, 2] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %120 = llvm.insertvalue %20, %119[3, 3] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %121 = llvm.insertvalue %11, %120[4, 3] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %122 = llvm.intr.stacksave : !llvm.ptr
    %123 = llvm.alloca %11 x !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> : (i64) -> !llvm.ptr
    llvm.store %103, %123 : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>, !llvm.ptr
    %124 = llvm.mlir.undef : !llvm.struct<(i64, ptr)>
    %125 = llvm.insertvalue %0, %124[0] : !llvm.struct<(i64, ptr)> 
    %126 = llvm.insertvalue %123, %125[1] : !llvm.struct<(i64, ptr)> 
    %127 = llvm.alloca %11 x !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> : (i64) -> !llvm.ptr
    llvm.store %121, %127 : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>, !llvm.ptr
    %128 = llvm.mlir.undef : !llvm.struct<(i64, ptr)>
    %129 = llvm.insertvalue %0, %128[0] : !llvm.struct<(i64, ptr)> 
    %130 = llvm.insertvalue %127, %129[1] : !llvm.struct<(i64, ptr)> 
    %131 = llvm.alloca %11 x !llvm.struct<(i64, ptr)> : (i64) -> !llvm.ptr
    llvm.store %126, %131 : !llvm.struct<(i64, ptr)>, !llvm.ptr
    %132 = llvm.alloca %11 x !llvm.struct<(i64, ptr)> : (i64) -> !llvm.ptr
    llvm.store %130, %132 : !llvm.struct<(i64, ptr)>, !llvm.ptr
    %133 = llvm.mlir.zero : !llvm.ptr
    %134 = llvm.getelementptr %133[1] : (!llvm.ptr) -> !llvm.ptr, f32
    %135 = llvm.ptrtoint %134 : !llvm.ptr to i64
    llvm.call @memrefCopy(%135, %131, %132) : (i64, !llvm.ptr, !llvm.ptr) -> ()
    llvm.intr.stackrestore %122 : !llvm.ptr
    %136 = llvm.add %73, %11  : i64
    llvm.br ^bb7(%136 : i64)
  ^bb9:  // pred: ^bb7
    %137 = llvm.add %71, %11  : i64
    llvm.br ^bb5(%137 : i64)
  ^bb10:  // pred: ^bb5
    %138 = llvm.add %69, %11  : i64
    llvm.br ^bb3(%138 : i64)
  ^bb11:  // pred: ^bb3
    %139 = llvm.add %67, %11  : i64
    llvm.br ^bb1(%139 : i64)
  ^bb12:  // pred: ^bb1
    llvm.call @free(%36) : (!llvm.ptr) -> ()
    llvm.call @free(%25) : (!llvm.ptr) -> ()
    %140 = llvm.call @nanoTime() : () -> i64
    %141 = llvm.sub %140, %66  : i64
    llvm.call @printI64(%141) : (i64) -> ()
    llvm.call @printNewline() : () -> ()
    llvm.return %65 : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
  }
  llvm.func @main() {
    %0 = llvm.call @matmul() : () -> !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
    llvm.return
  }
}

