// RUN: mlir-opt -split-input-file -test-tensor-transform-patterns=test-fold-pack-and-unpack %s | FileCheck %s

// TODO: Trivial case
//       Q: Is it fully safe to do?
//       Q: Is it needed/beneficial?
func.func @fold_unpack_into_parallel_insert_slice_into_empty(
    %arg0: tensor<32x32xbf16>, %arg1: tensor<64x64xbf16>, %x: index, %y: index) -> tensor<64x64xbf16> {
  %e = tensor.empty() : tensor<2x2x32x32xbf16>
  %0 = scf.forall (%i, %j) in (%x, %y) shared_outs(%out = %e) -> (tensor<2x2x32x32xbf16>) {
    scf.forall.in_parallel {
      tensor.parallel_insert_slice %arg0 into %out[%i, %j, 0, 0] [1, 1, 32, 32] [1, 1, 1, 1] : tensor<32x32xbf16> into tensor<2x2x32x32xbf16>
    }
  }
  %unpack = tensor.unpack %0 inner_dims_pos = [0, 1] inner_tiles = [32, 32] into %arg1
    : tensor<2x2x32x32xbf16> -> tensor<64x64xbf16>
  return %unpack : tensor<64x64xbf16>
}

// CHECK-DAG: #[[MAP:.+]] = affine_map<(d0) -> (d0 * 32)>

// CHECK-LABEL: @fold_unpack_into_parallel_insert_slice_into_producer
// CHECK-SAME: %[[ARG0:.+]]: tensor<32x32xbf16>, %[[ARG1:.+]]: tensor<64x64xbf16>,
// CHECK-SAME: %[[X:.+]]: index, %[[Y:.+]]: index
// CHECK: %[[EMPTY:.+]] = tensor.empty() : tensor<2x2x32x32xbf16>
// CHECK: scf.forall (%[[I:.+]], %[[J:.+]]) in (%[[X]], %[[Y]])
// CHECK-SAME: shared_outs(%[[OLD_OUT:.+]] = %[[EMPTY]], %[[NEW_OUT:.+]] = %[[ARG1]])
// CHECK: %[[AFFINE_I:.+]] = affine.apply #[[MAP]](%[[I]])
// CHECK: %[[AFFINE_J:.+]] = affine.apply #[[MAP]](%[[J]])
// CHECK: tensor.parallel_insert_slice %[[ARG0]] into %[[NEW_OUT]][%[[AFFINE_I]], %[[AFFINE_J]]] [32, 32] [1, 1] : tensor<32x32xbf16> into tensor<64x64xbf16>
// CHECK-NOT: tensor.unpack

// -----

func.func @fold_unpack_into_parallel_insert_slice_into_producer(
    %arg0: tensor<64x64xbf16>, %arg1: tensor<32x32xbf16>, %arg2: tensor<64x64xbf16>, %x: index, %y: index) -> tensor<64x64xbf16> {
  %packed_layout = tensor.empty() : tensor<2x2x32x32xbf16>
  %pack = tensor.pack %arg0 inner_dims_pos = [0, 1] inner_tiles = [32, 32] into %packed_layout
    : tensor<64x64xbf16> -> tensor<2x2x32x32xbf16>
  %0 = scf.forall (%i, %j) in (%x, %y) shared_outs(%out = %pack) -> (tensor<2x2x32x32xbf16>) {
    scf.forall.in_parallel {
      tensor.parallel_insert_slice %arg1 into %out[%i, %j, 0, 0] [1, 1, 32, 32] [1, 1, 1, 1] : tensor<32x32xbf16> into tensor<2x2x32x32xbf16>
    }
  }
  %unpack = tensor.unpack %0 inner_dims_pos = [0, 1] inner_tiles = [32, 32] into %arg2
    : tensor<2x2x32x32xbf16> -> tensor<64x64xbf16>
  return %unpack : tensor<64x64xbf16>
}

// CHECK-DAG: #[[MAP:.+]] = affine_map<(d0) -> (d0 * 32)>

// CHECK-LABEL: @fold_unpack_into_parallel_insert_slice_into_producer
// CHECK-SAME: %[[ARG0:.+]]: tensor<64x64xbf16>, %[[ARG1:.+]]: tensor<32x32xbf16>,
// CHECK-SAME: %[[ARG2:.+]]: tensor<64x64xbf16>,
// CHECK-SAME: %[[X:.+]]: index, %[[Y:.+]]: index
// CHECK: %[[PRODUCER]] = tensor.pack %[[ARG0]]
// CHECK: scf.forall (%[[I:.+]], %[[J:.+]]) in (%[[X]], %[[Y]])
// CHECK-SAME: shared_outs(%[[OLD_OUT:.+]] = %[[PRODUCER]], %[[NEW_OUT:.+]] = %[[ARG2]])
// CHECK: %[[AFFINE_I:.+]] = affine.apply #[[MAP]](%[[I]])
// CHECK: %[[AFFINE_J:.+]] = affine.apply #[[MAP]](%[[J]])
// CHECK: tensor.parallel_insert_slice %[[ARG1]] into %[[NEW_OUT]][%[[AFFINE_I]], %[[AFFINE_J]]] [32, 32] [1, 1] : tensor<32x32xbf16> into tensor<64x64xbf16>
// CHECK-NOT: tensor.unpack

// -----

// TODO: Should this be folded?
//       If yes, fix current folder to work here.
func.func @fold_unpack_into_parallel_insert_slice_partial(
    %arg0: tensor<32x32xbf16>, %arg1: tensor<64x64xbf16>, %x: index) -> tensor<64x64xbf16> {
  %e = tensor.empty() : tensor<2x2x32x32xbf16>
  %0 = scf.forall (%i) in (%x) shared_outs(%out = %e) -> (tensor<2x2x32x32xbf16>) {
    scf.forall.in_parallel {
      tensor.parallel_insert_slice %arg0 into %out[%i, 0, 0, 0] [1, 1, 32, 32] [1, 1, 1, 1] : tensor<32x32xbf16> into tensor<2x2x32x32xbf16>
    }
  }
  %unpack = tensor.unpack %0 inner_dims_pos = [0, 1] inner_tiles = [32, 32] into %arg1
    : tensor<2x2x32x32xbf16> -> tensor<64x64xbf16>
  return %unpack : tensor<64x64xbf16>
}

// Negative test with an unsupported unpack op, should be a no op.

// CHECK-LABEL: @expected_failure_unpack_with_outer_dims_perm

func.func @expected_failure_unpack_with_outer_dims_perm(
    %arg0: tensor<64x64xbf16>, %arg1: tensor<32x32xbf16>, %arg2: tensor<64x64xbf16>, %x: index, %y: index) -> tensor<64x64xbf16> {
  %packed_layout = tensor.empty() : tensor<2x2x32x32xbf16>
  %pack = tensor.pack %arg0 inner_dims_pos = [0, 1] inner_tiles = [32, 32] into %packed_layout
    : tensor<64x64xbf16> -> tensor<2x2x32x32xbf16>
  %0 = scf.forall (%i, %j) in (%x, %y) shared_outs(%out = %pack) -> (tensor<2x2x32x32xbf16>) {
    scf.forall.in_parallel {
      tensor.parallel_insert_slice %arg1 into %out[%i, %j, 0, 0] [1, 1, 32, 32] [1, 1, 1, 1] : tensor<32x32xbf16> into tensor<2x2x32x32xbf16>
    }
  }
  %unpack = tensor.unpack %0 outer_dims_perm = [1, 0] inner_dims_pos = [0, 1] inner_tiles = [32, 32] into %arg2
    : tensor<2x2x32x32xbf16> -> tensor<64x64xbf16>
  return %unpack : tensor<64x64xbf16>
}


// Negative test with an unsupported unsupported parallel_insert_slice, should be a no op.

// CHECK-LABEL: @expected_failure_for_unsupported_parallel_insert_slice_indexing1

func.func @expected_failure_for_unsupported_parallel_insert_slice_indexing1(
    %arg0: tensor<64x64xbf16>, %arg1: tensor<32x32xbf16>, %arg2: tensor<64x64xbf16>, %x: index, %y: index) -> tensor<64x64xbf16> {
  %packed_layout = tensor.empty() : tensor<2x2x32x32xbf16>
  %pack = tensor.pack %arg0 inner_dims_pos = [0, 1] inner_tiles = [32, 32] into %packed_layout
    : tensor<64x64xbf16> -> tensor<2x2x32x32xbf16>
  %0 = scf.forall (%i, %j) in (%x, %y) shared_outs(%out = %pack) -> (tensor<2x2x32x32xbf16>) {
    scf.forall.in_parallel {
      tensor.parallel_insert_slice %arg1 into %out[%i, 0, 0, 0] [1, 1, 32, 32] [1, 1, 1, 1] : tensor<32x32xbf16> into tensor<2x2x32x32xbf16>
    }
  }
  %unpack = tensor.unpack %0 inner_dims_pos = [0, 1] inner_tiles = [32, 32] into %arg2
    : tensor<2x2x32x32xbf16> -> tensor<64x64xbf16>
  return %unpack : tensor<64x64xbf16>
}

// Negative test with an unsupported unsupported parallel_insert_slice, should be a no op.

// CHECK-LABEL: @expected_failure_for_unsupported_parallel_insert_slice_indexing2
func.func @expected_failure_for_unsupported_parallel_insert_slice_indexing2(
    %arg0: tensor<64x64xbf16>, %arg1: tensor<32x32xbf16>, %arg2: tensor<64x64xbf16>, %x: index, %y: index) -> tensor<64x64xbf16> {
  %packed_layout = tensor.empty() : tensor<2x2x32x32xbf16>
  %pack = tensor.pack %arg0 inner_dims_pos = [0, 1] inner_tiles = [32, 32] into %packed_layout
    : tensor<64x64xbf16> -> tensor<2x2x32x32xbf16>
  %0 = scf.forall (%i, %j) in (%x, %y) shared_outs(%out = %pack) -> (tensor<2x2x32x32xbf16>) {
    scf.forall.in_parallel {
      tensor.parallel_insert_slice %arg1 into %out[%j, %i, 0, 0] [1, 1, 32, 32] [1, 1, 1, 1] : tensor<32x32xbf16> into tensor<2x2x32x32xbf16>
    }
  }
  %unpack = tensor.unpack %0 inner_dims_pos = [0, 1] inner_tiles = [32, 32] into %arg2
    : tensor<2x2x32x32xbf16> -> tensor<64x64xbf16>
  return %unpack : tensor<64x64xbf16>
}

// CHECK-LABEL: @fold_unpack_into_parallel_insert_slice_partial
// TODO: Add expected IR checks.

// -----

// TODO: Add test case when there's a producer inside the loop which result gets inserted.

// -----

// TODO: Adds test case with multiple users of scf.forall result.

// -----

// TODO: This case probably shouldn't be folded.
//       Is it safe to fold when `shared_outs` is directly an argument?
//       - Q: Fold when insert performed on the argument?
//       - Q: Fold when insert and unpack target two different arguments?
func.func @no_fold_unpack_into_parallel_insert_slice(
    %arg0: tensor<2x2x32x32xbf16>, %arg1: tensor<32x32xbf16>, %arg2: tensor<64x64xbf16>, %x: index, %y: index) -> tensor<64x64xbf16> {
  %0 = scf.forall (%i, %j) in (%x, %y) shared_outs(%out = %arg0) -> (tensor<2x2x32x32xbf16>) {
    scf.forall.in_parallel {
      tensor.parallel_insert_slice %arg1 into %out[%i, %j, 0, 0] [1, 1, 32, 32] [1, 1, 1, 1] : tensor<32x32xbf16> into tensor<2x2x32x32xbf16>
    }
  }
  %unpack = tensor.unpack %0 inner_dims_pos = [0, 1] inner_tiles = [32, 32] into %arg2
    : tensor<2x2x32x32xbf16> -> tensor<64x64xbf16>
  return %unpack : tensor<64x64xbf16>
}

// CHECK-LABEL: @no_fold_unpack_in_parallel_insert_slice
// CHECK: scf.forall
// CHECK: tensor.unpack
