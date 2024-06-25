//===- FoldUnPackIntoInsertSlice.cpp ---------------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
// Implements a rewrite pattern to fold a tensor.unpack into an
// scf.parallel_insert.
//
// The pattern looks like:
//
// %p = tensor.pack %a into %b
// %l = scf.forall ... iter_args(%0 = %p) {
// ...
// }
// %u = tensor.unpack %l into %c
//
// We will rewrite as:
//
// %l = scf.forall ... iter_args(%0 = %a) {
// ...
// }
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/Utils.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tensor/Utils/Utils.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/SmallVector.h"

using namespace mlir;

struct FoldUnPackIntoInsertSlice : public OpRewritePattern<tensor::UnPackOp> {
  using OpRewritePattern<tensor::UnPackOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(tensor::UnPackOp unPackOp,
                                PatternRewriter &rewriter) const override {
    if (!unPackOp.getOuterDimsPerm().empty())
      return failure();
    SmallVector<int64_t> innerDimsPos =
        llvm::to_vector(unPackOp.getInnerDimsPos());
    SmallVector<int64_t> expectedDimsPos = llvm::to_vector(
        llvm::seq<int64_t>(0, unPackOp.getDestType().getRank()));
    if (innerDimsPos != expectedDimsPos)
      return failure();

    Operation *loop = unPackOp.getSource().getDefiningOp();
    if (!isa_and_nonnull<scf::ForallOp>(loop))
      return failure();
    auto forallOp = cast<scf::ForallOp>(loop);
    if (!forallOp->hasOneUse() || forallOp->getNumResults() != 1)
      return failure();
    OpBuilder::InsertionGuard g(rewriter);
    rewriter.setInsertionPoint(forallOp);

    // Create a new scf.forall operation, updating its output.
    Value loopOperand =
        forallOp.getTiedOpOperand(forallOp->getResult(0))->get();
    tensor::PackOp packOp =
        dyn_cast_or_null<tensor::PackOp>(loopOperand.getDefiningOp());
    if (!packOp)
      return failure();
    Value newLoopOperand = packOp.getSource();
    SmallVector<Value> newOuts(forallOp.getOutputs());
    if (newOuts.size() != 1)
      return failure();

    newOuts.push_back(newLoopOperand);
    auto newForallOp = rewriter.create<scf::ForallOp>(
        forallOp.getLoc(), forallOp.getMixedLowerBound(),
        forallOp.getMixedUpperBound(), forallOp.getMixedStep(), newOuts,
        forallOp.getMapping());
    rewriter.eraseBlock(newForallOp.getBody());
    newForallOp.getRegion().takeBody(forallOp.getRegion());
    newForallOp.getBody()->addArgument(newOuts.back().getType(),
                                       newOuts.back().getLoc());

    ArrayRef<BlockArgument> bbArgs = newForallOp.getRegionIterArgs();
    assert(bbArgs.size() == 2);

    rewriter.setInsertionPointToStart(newForallOp.getBody());
    AffineExpr dim0;
    bindDims(rewriter.getContext(), dim0);
    AffineExpr s0 = rewriter.getAffineSymbolExpr(0);
    auto mulMap = AffineMap::get(1, 1, {dim0 * s0});
    SmallVector<OpFoldResult> newMixedOffsets;
    for (auto ivs : llvm::enumerate(newForallOp.getInductionVars())) {
      OpFoldResult applied = affine::makeComposedFoldedAffineApply(
          rewriter, newForallOp.getLoc(), mulMap,
          {ivs.value(), unPackOp.getMixedTiles()[ivs.index()]});
      newMixedOffsets.push_back(applied);
    }

    for (Operation *operation : bbArgs.front().getUsers()) {
      if (auto extractSliceOp = dyn_cast<tensor::ExtractSliceOp>(operation)) {
        rewriter.setInsertionPoint(extractSliceOp);

        int64_t rank = unPackOp.getDestType().getRank();
        auto mixedStrides = extractSliceOp.getMixedStrides();
        auto newMixedStrides = SmallVector<OpFoldResult>(
            mixedStrides.begin() + rank, mixedStrides.end());

        auto mixedSizes = extractSliceOp.getMixedSizes();
        auto newMixedSizes = SmallVector<OpFoldResult>(
            mixedSizes.begin() + rank, mixedSizes.end());

        auto newExtractSliceOp = rewriter.create<tensor::ExtractSliceOp>(
            extractSliceOp.getLoc(), bbArgs.back(), newMixedOffsets,
            newMixedSizes, newMixedStrides);

        rewriter.replaceAllUsesWith(extractSliceOp->getResults(),
                                    newExtractSliceOp->getResults());
        continue;
      }
      if (auto parallelInsertSlice =
              dyn_cast<tensor::ParallelInsertSliceOp>(operation)) {
        rewriter.setInsertionPoint(parallelInsertSlice);

        int64_t rank = unPackOp.getDestType().getRank();
        auto mixedStrides = parallelInsertSlice.getMixedStrides();
        auto newMixedStrides = SmallVector<OpFoldResult>(
            mixedStrides.begin() + rank, mixedStrides.end());

        auto mixedSizes = parallelInsertSlice.getMixedSizes();
        auto newMixedSizes = SmallVector<OpFoldResult>(
            mixedSizes.begin() + rank, mixedSizes.end());

        auto newInsertSliceOp = rewriter.create<tensor::ParallelInsertSliceOp>(
            parallelInsertSlice.getLoc(), parallelInsertSlice.getSource(),
            bbArgs.back(), newMixedOffsets, newMixedSizes, newMixedStrides);
        rewriter.replaceAllUsesWith(parallelInsertSlice->getResults(),
                                    newInsertSliceOp->getResults());
        rewriter.eraseOp(parallelInsertSlice);
        continue;
      }
      return failure();
    }

    rewriter.replaceOp(unPackOp, newForallOp->getResults()[1]);
    return success();
  }
};
