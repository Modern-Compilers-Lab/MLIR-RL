module attributes {llvm.data_layout = ""} {
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
  llvm.func @printMemrefF32(i64, !llvm.ptr) attributes {sym_visibility = "private"}
  llvm.func @matmul() -> !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> {
    %0 = llvm.mlir.constant(4 : index) : i64
    %1 = llvm.mlir.constant(2 : i64) : i64
    %2 = llvm.mlir.constant(64 : index) : i64
    %3 = llvm.mlir.constant(1200 : index) : i64
    %4 = llvm.mlir.constant(5 : index) : i64
    %5 = llvm.mlir.constant(1000 : index) : i64
    %6 = llvm.mlir.constant(dense<0.000000e+00> : vector<5x1000xf32>) : !llvm.array<5 x vector<1000xf32>>
    %7 = llvm.mlir.constant(dense<2.000000e+00> : vector<1500x1000xf32>) : !llvm.array<1500 x vector<1000xf32>>
    %8 = llvm.mlir.constant(dense<2.000000e+00> : vector<5x1500xf32>) : !llvm.array<5 x vector<1500xf32>>
    %9 = llvm.mlir.constant(1500 : index) : i64
    %10 = llvm.mlir.constant(0 : index) : i64
    %11 = llvm.mlir.constant(240 : index) : i64
    %12 = llvm.mlir.constant(1 : index) : i64
    %13 = llvm.mlir.null : !llvm.ptr
    %14 = llvm.getelementptr %13[1800000] : (!llvm.ptr) -> !llvm.ptr, f32
    %15 = llvm.ptrtoint %14 : !llvm.ptr to i64
    %16 = llvm.add %15, %2  : i64
    %17 = llvm.call @malloc(%16) : (i64) -> !llvm.ptr
    %18 = llvm.ptrtoint %17 : !llvm.ptr to i64
    %19 = llvm.sub %2, %12  : i64
    %20 = llvm.add %18, %19  : i64
    %21 = llvm.urem %20, %2  : i64
    %22 = llvm.sub %20, %21  : i64
    %23 = llvm.inttoptr %22 : i64 to !llvm.ptr
    %24 = llvm.mlir.null : !llvm.ptr
    %25 = llvm.getelementptr %24[1500000] : (!llvm.ptr) -> !llvm.ptr, f32
    %26 = llvm.ptrtoint %25 : !llvm.ptr to i64
    %27 = llvm.add %26, %2  : i64
    %28 = llvm.call @malloc(%27) : (i64) -> !llvm.ptr
    %29 = llvm.ptrtoint %28 : !llvm.ptr to i64
    %30 = llvm.sub %2, %12  : i64
    %31 = llvm.add %29, %30  : i64
    %32 = llvm.urem %31, %2  : i64
    %33 = llvm.sub %31, %32  : i64
    %34 = llvm.inttoptr %33 : i64 to !llvm.ptr
    %35 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %36 = llvm.insertvalue %28, %35[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %37 = llvm.insertvalue %34, %36[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %38 = llvm.insertvalue %10, %37[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %39 = llvm.insertvalue %9, %38[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %40 = llvm.insertvalue %5, %39[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %41 = llvm.insertvalue %5, %40[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %42 = llvm.insertvalue %12, %41[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %43 = llvm.mlir.null : !llvm.ptr
    %44 = llvm.getelementptr %43[1200000] : (!llvm.ptr) -> !llvm.ptr, f32
    %45 = llvm.ptrtoint %44 : !llvm.ptr to i64
    %46 = llvm.add %45, %2  : i64
    %47 = llvm.call @malloc(%46) : (i64) -> !llvm.ptr
    %48 = llvm.ptrtoint %47 : !llvm.ptr to i64
    %49 = llvm.sub %2, %12  : i64
    %50 = llvm.add %48, %49  : i64
    %51 = llvm.urem %50, %2  : i64
    %52 = llvm.sub %50, %51  : i64
    %53 = llvm.inttoptr %52 : i64 to !llvm.ptr
    %54 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %55 = llvm.insertvalue %47, %54[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %56 = llvm.insertvalue %53, %55[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %57 = llvm.insertvalue %10, %56[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %58 = llvm.insertvalue %3, %57[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %59 = llvm.insertvalue %5, %58[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %60 = llvm.insertvalue %5, %59[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %61 = llvm.insertvalue %12, %60[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %62 = llvm.call @nanoTime() : () -> i64
    omp.parallel   {
      %65 = llvm.alloca %12 x !llvm.array<5 x vector<1500xf32>> : (i64) -> !llvm.ptr
      %66 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64)>
      %67 = llvm.insertvalue %65, %66[0] : !llvm.struct<(ptr, ptr, i64)> 
      %68 = llvm.insertvalue %65, %67[1] : !llvm.struct<(ptr, ptr, i64)> 
      %69 = llvm.insertvalue %10, %68[2] : !llvm.struct<(ptr, ptr, i64)> 
      %70 = llvm.alloca %12 x !llvm.array<1500 x vector<1000xf32>> : (i64) -> !llvm.ptr
      %71 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64)>
      %72 = llvm.insertvalue %70, %71[0] : !llvm.struct<(ptr, ptr, i64)> 
      %73 = llvm.insertvalue %70, %72[1] : !llvm.struct<(ptr, ptr, i64)> 
      %74 = llvm.insertvalue %10, %73[2] : !llvm.struct<(ptr, ptr, i64)> 
      %75 = llvm.alloca %12 x !llvm.array<5 x vector<1000xf32>> : (i64) -> !llvm.ptr
      %76 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64)>
      %77 = llvm.insertvalue %75, %76[0] : !llvm.struct<(ptr, ptr, i64)> 
      %78 = llvm.insertvalue %75, %77[1] : !llvm.struct<(ptr, ptr, i64)> 
      %79 = llvm.insertvalue %10, %78[2] : !llvm.struct<(ptr, ptr, i64)> 
      omp.wsloop   for  (%arg0) : i64 = (%10) to (%11) step (%12) {
        %80 = llvm.mul %arg0, %4  : i64
        %81 = llvm.mul %80, %9  : i64
        %82 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
        %83 = llvm.insertvalue %17, %82[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
        %84 = llvm.insertvalue %23, %83[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
        %85 = llvm.insertvalue %81, %84[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
        %86 = llvm.insertvalue %4, %85[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
        %87 = llvm.insertvalue %9, %86[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
        %88 = llvm.insertvalue %9, %87[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
        %89 = llvm.insertvalue %12, %88[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
        llvm.store %8, %65 : !llvm.array<5 x vector<1500xf32>>, !llvm.ptr
        %90 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
        %91 = llvm.insertvalue %65, %90[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
        %92 = llvm.insertvalue %65, %91[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
        %93 = llvm.mlir.constant(0 : index) : i64
        %94 = llvm.insertvalue %93, %92[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
        %95 = llvm.mlir.constant(5 : index) : i64
        %96 = llvm.insertvalue %95, %94[3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
        %97 = llvm.mlir.constant(1 : index) : i64
        %98 = llvm.insertvalue %97, %96[4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
        llvm.br ^bb1(%10 : i64)
      ^bb1(%99: i64):  // 2 preds: ^bb0, ^bb2
        %100 = llvm.icmp "slt" %99, %4 : i64
        llvm.cond_br %100, ^bb2, ^bb3
      ^bb2:  // pred: ^bb1
        %101 = llvm.extractvalue %98[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
        %102 = llvm.getelementptr %101[%99] : (!llvm.ptr, i64) -> !llvm.ptr, vector<1500xf32>
        %103 = llvm.load %102 : !llvm.ptr -> vector<1500xf32>
        %104 = llvm.getelementptr %23[%81] : (!llvm.ptr, i64) -> !llvm.ptr, f32
        %105 = llvm.mlir.constant(1500 : index) : i64
        %106 = llvm.mul %99, %105  : i64
        %107 = llvm.add %106, %10  : i64
        %108 = llvm.getelementptr %104[%107] : (!llvm.ptr, i64) -> !llvm.ptr, f32
        llvm.store %103, %108 {alignment = 4 : i64} : vector<1500xf32>, !llvm.ptr
        %109 = llvm.add %99, %12  : i64
        llvm.br ^bb1(%109 : i64)
      ^bb3:  // pred: ^bb1
        llvm.store %7, %70 : !llvm.array<1500 x vector<1000xf32>>, !llvm.ptr
        %110 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
        %111 = llvm.insertvalue %70, %110[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
        %112 = llvm.insertvalue %70, %111[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
        %113 = llvm.mlir.constant(0 : index) : i64
        %114 = llvm.insertvalue %113, %112[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
        %115 = llvm.mlir.constant(1500 : index) : i64
        %116 = llvm.insertvalue %115, %114[3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
        %117 = llvm.mlir.constant(1 : index) : i64
        %118 = llvm.insertvalue %117, %116[4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
        llvm.br ^bb4(%10 : i64)
      ^bb4(%119: i64):  // 2 preds: ^bb3, ^bb5
        %120 = llvm.icmp "slt" %119, %9 : i64
        llvm.cond_br %120, ^bb5, ^bb6
      ^bb5:  // pred: ^bb4
        %121 = llvm.extractvalue %118[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
        %122 = llvm.getelementptr %121[%119] : (!llvm.ptr, i64) -> !llvm.ptr, vector<1000xf32>
        %123 = llvm.load %122 : !llvm.ptr -> vector<1000xf32>
        %124 = llvm.mlir.constant(1000 : index) : i64
        %125 = llvm.mul %119, %124  : i64
        %126 = llvm.add %125, %10  : i64
        %127 = llvm.getelementptr %34[%126] : (!llvm.ptr, i64) -> !llvm.ptr, f32
        llvm.store %123, %127 {alignment = 4 : i64} : vector<1000xf32>, !llvm.ptr
        %128 = llvm.add %119, %12  : i64
        llvm.br ^bb4(%128 : i64)
      ^bb6:  // pred: ^bb4
        %129 = llvm.mul %80, %5  : i64
        %130 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
        %131 = llvm.insertvalue %47, %130[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
        %132 = llvm.insertvalue %53, %131[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
        %133 = llvm.insertvalue %129, %132[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
        %134 = llvm.insertvalue %4, %133[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
        %135 = llvm.insertvalue %5, %134[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
        %136 = llvm.insertvalue %5, %135[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
        %137 = llvm.insertvalue %12, %136[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
        llvm.store %6, %75 : !llvm.array<5 x vector<1000xf32>>, !llvm.ptr
        %138 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
        %139 = llvm.insertvalue %75, %138[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
        %140 = llvm.insertvalue %75, %139[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
        %141 = llvm.mlir.constant(0 : index) : i64
        %142 = llvm.insertvalue %141, %140[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
        %143 = llvm.mlir.constant(5 : index) : i64
        %144 = llvm.insertvalue %143, %142[3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
        %145 = llvm.mlir.constant(1 : index) : i64
        %146 = llvm.insertvalue %145, %144[4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
        llvm.br ^bb7(%10 : i64)
      ^bb7(%147: i64):  // 2 preds: ^bb6, ^bb8
        %148 = llvm.icmp "slt" %147, %4 : i64
        llvm.cond_br %148, ^bb8, ^bb9
      ^bb8:  // pred: ^bb7
        %149 = llvm.extractvalue %146[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
        %150 = llvm.getelementptr %149[%147] : (!llvm.ptr, i64) -> !llvm.ptr, vector<1000xf32>
        %151 = llvm.load %150 : !llvm.ptr -> vector<1000xf32>
        %152 = llvm.getelementptr %53[%129] : (!llvm.ptr, i64) -> !llvm.ptr, f32
        %153 = llvm.mlir.constant(1000 : index) : i64
        %154 = llvm.mul %147, %153  : i64
        %155 = llvm.add %154, %10  : i64
        %156 = llvm.getelementptr %152[%155] : (!llvm.ptr, i64) -> !llvm.ptr, f32
        llvm.store %151, %156 {alignment = 4 : i64} : vector<1000xf32>, !llvm.ptr
        %157 = llvm.add %147, %12  : i64
        llvm.br ^bb7(%157 : i64)
      ^bb9:  // pred: ^bb7
        llvm.br ^bb10(%10 : i64)
      ^bb10(%158: i64):  // 2 preds: ^bb9, ^bb20
        %159 = llvm.icmp "slt" %158, %5 : i64
        llvm.cond_br %159, ^bb11, ^bb21
      ^bb11:  // pred: ^bb10
        %160 = llvm.add %129, %158  : i64
        %161 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
        %162 = llvm.insertvalue %47, %161[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
        %163 = llvm.insertvalue %53, %162[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
        %164 = llvm.insertvalue %160, %163[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
        %165 = llvm.insertvalue %4, %164[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
        %166 = llvm.insertvalue %5, %165[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
        %167 = llvm.insertvalue %4, %166[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
        %168 = llvm.insertvalue %12, %167[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
        llvm.br ^bb12(%10 : i64)
      ^bb12(%169: i64):  // 2 preds: ^bb11, ^bb19
        %170 = llvm.icmp "slt" %169, %4 : i64
        llvm.cond_br %170, ^bb13, ^bb20
      ^bb13:  // pred: ^bb12
        llvm.br ^bb14(%10 : i64)
      ^bb14(%171: i64):  // 2 preds: ^bb13, ^bb18
        %172 = llvm.icmp "slt" %171, %4 : i64
        llvm.cond_br %172, ^bb15, ^bb19
      ^bb15:  // pred: ^bb14
        llvm.br ^bb16(%10 : i64)
      ^bb16(%173: i64):  // 2 preds: ^bb15, ^bb17
        %174 = llvm.icmp "slt" %173, %9 : i64
        llvm.cond_br %174, ^bb17, ^bb18
      ^bb17:  // pred: ^bb16
        %175 = llvm.getelementptr %23[%81] : (!llvm.ptr, i64) -> !llvm.ptr, f32
        %176 = llvm.mul %169, %9  : i64
        %177 = llvm.add %176, %173  : i64
        %178 = llvm.getelementptr %175[%177] : (!llvm.ptr, i64) -> !llvm.ptr, f32
        %179 = llvm.load %178 : !llvm.ptr -> f32
        %180 = llvm.getelementptr %34[%158] : (!llvm.ptr, i64) -> !llvm.ptr, f32
        %181 = llvm.mul %173, %5  : i64
        %182 = llvm.add %181, %171  : i64
        %183 = llvm.getelementptr %180[%182] : (!llvm.ptr, i64) -> !llvm.ptr, f32
        %184 = llvm.load %183 : !llvm.ptr -> f32
        %185 = llvm.getelementptr %53[%160] : (!llvm.ptr, i64) -> !llvm.ptr, f32
        %186 = llvm.mul %169, %5  : i64
        %187 = llvm.add %186, %171  : i64
        %188 = llvm.getelementptr %185[%187] : (!llvm.ptr, i64) -> !llvm.ptr, f32
        %189 = llvm.load %188 : !llvm.ptr -> f32
        %190 = llvm.fmul %179, %184  : f32
        %191 = llvm.fadd %189, %190  : f32
        %192 = llvm.getelementptr %53[%160] : (!llvm.ptr, i64) -> !llvm.ptr, f32
        %193 = llvm.mul %169, %5  : i64
        %194 = llvm.add %193, %171  : i64
        %195 = llvm.getelementptr %192[%194] : (!llvm.ptr, i64) -> !llvm.ptr, f32
        llvm.store %191, %195 : f32, !llvm.ptr
        %196 = llvm.add %173, %12  : i64
        llvm.br ^bb16(%196 : i64)
      ^bb18:  // pred: ^bb16
        %197 = llvm.add %171, %12  : i64
        llvm.br ^bb14(%197 : i64)
      ^bb19:  // pred: ^bb14
        %198 = llvm.add %169, %12  : i64
        llvm.br ^bb12(%198 : i64)
      ^bb20:  // pred: ^bb12
        %199 = llvm.add %129, %158  : i64
        %200 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
        %201 = llvm.insertvalue %47, %200[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
        %202 = llvm.insertvalue %53, %201[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
        %203 = llvm.insertvalue %199, %202[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
        %204 = llvm.insertvalue %4, %203[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
        %205 = llvm.insertvalue %5, %204[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
        %206 = llvm.insertvalue %4, %205[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
        %207 = llvm.insertvalue %12, %206[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
        %208 = llvm.intr.stacksave : !llvm.ptr
        %209 = llvm.alloca %12 x !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> : (i64) -> !llvm.ptr
        llvm.store %168, %209 : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>, !llvm.ptr
        %210 = llvm.mlir.undef : !llvm.struct<(i64, ptr)>
        %211 = llvm.insertvalue %1, %210[0] : !llvm.struct<(i64, ptr)> 
        %212 = llvm.insertvalue %209, %211[1] : !llvm.struct<(i64, ptr)> 
        %213 = llvm.alloca %12 x !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> : (i64) -> !llvm.ptr
        llvm.store %207, %213 : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>, !llvm.ptr
        %214 = llvm.mlir.undef : !llvm.struct<(i64, ptr)>
        %215 = llvm.insertvalue %1, %214[0] : !llvm.struct<(i64, ptr)> 
        %216 = llvm.insertvalue %213, %215[1] : !llvm.struct<(i64, ptr)> 
        %217 = llvm.alloca %12 x !llvm.struct<(i64, ptr)> : (i64) -> !llvm.ptr
        llvm.store %212, %217 : !llvm.struct<(i64, ptr)>, !llvm.ptr
        %218 = llvm.alloca %12 x !llvm.struct<(i64, ptr)> : (i64) -> !llvm.ptr
        llvm.store %216, %218 : !llvm.struct<(i64, ptr)>, !llvm.ptr
        llvm.call @memrefCopy(%0, %217, %218) : (i64, !llvm.ptr, !llvm.ptr) -> ()
        llvm.intr.stackrestore %208 : !llvm.ptr
        %219 = llvm.add %158, %4  : i64
        llvm.br ^bb10(%219 : i64)
      ^bb21:  // pred: ^bb10
        %220 = llvm.mul %80, %5  : i64
        %221 = llvm.mul %4, %12  : i64
        %222 = llvm.mul %221, %5  : i64
        %223 = llvm.mlir.null : !llvm.ptr
        %224 = llvm.getelementptr %223[1] : (!llvm.ptr) -> !llvm.ptr, f32
        %225 = llvm.ptrtoint %224 : !llvm.ptr to i64
        %226 = llvm.mul %222, %225  : i64
        %227 = llvm.getelementptr %53[%129] : (!llvm.ptr, i64) -> !llvm.ptr, f32
        %228 = llvm.getelementptr %53[%220] : (!llvm.ptr, i64) -> !llvm.ptr, f32
        "llvm.intr.memcpy"(%228, %227, %226) <{isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i64) -> ()
        omp.yield
      }
      omp.terminator
    }
    %63 = llvm.call @nanoTime() : () -> i64
    %64 = llvm.sub %63, %62  : i64
    llvm.call @printI64(%64) : (i64) -> ()
    llvm.call @free(%17) : (!llvm.ptr) -> ()
    llvm.call @free(%28) : (!llvm.ptr) -> ()
    llvm.return %61 : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
  }
  llvm.func @main() {
    %0 = llvm.call @matmul() : () -> !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    llvm.return
  }
}

