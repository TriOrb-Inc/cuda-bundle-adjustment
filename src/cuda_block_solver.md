# cuda_block_solver.cu

## 目的

`cuda_block_solver.cu` は `cuda-bundle-adjustment` の GPU kernel 群と、Schur complement 前後で使う Thrust ベースの補助処理をまとめた source です。SLAM 本体から見て、CUDA BA の実計算をどこで担っているかを追いやすくするための sidecar として扱います。

## 対象範囲

- 対象 source: `slam-core/3rd/cuda-bundle-adjustment/src/cuda_block_solver.cu`
- 主な責務: active error 計算、Schur complement 構築、行列並べ替え、pose / landmark 更新
- 関連 entry point: `computeActiveErrors*`、`computeChiSquares*`、`computeBschure`、`computeHschure`、`schurComplementPost`

## 現状

- CUDA kernel と host helper が同じ translation unit にあり、Thrust の `exclusive_scan` / `sort` / `gather` もこの file で呼びます。
- `slam-core` 側 wrapper は solve 入口で `cudaSetDevice(0)` を呼びますが、Jetson AGX Thor のように driver / runtime の device context が不安定な host では、Thrust 側の host function でも thread-local device を握り直さないと `cudaErrorInvalidDevice` へ落ちることがあります。

## 実装上の判断

- この repo では `prepareCudaThreadContext()` を追加し、Thrust 呼び出しや主要な host 側 CUDA entry point の直前で `cudaSetDevice(0)` を再実行します。
- これは Thor で `parallel_for failed: cudaErrorInvalidDevice: invalid device ordinal` が出る経路を抑えるための局所対策です。
- wrapper 側の prewarm と重複して見えても、submodule 内の host helper が別のタイミングで実行される以上、この file 側でも context を明示する方を優先します。

## 目標

- Jetson Orin / Thor を含む複数世代の Jetson で、CUDA BA の host helper が device ordinal 不整合で落ちない状態を保つ。
- CUDA kernel 自体の問題と、host 側 CUDA context の問題を切り分けやすくする。

## 関連

- `slam-core/3rd/cuda-bundle-adjustment/src/cuda_block_solver.cu`
- `slam-core/3rd/cuda-bundle-adjustment/src/cuda_bundle_adjustment.cpp`
- `slam-core/3rd/cuda-bundle-adjustment/src/cuda_bundle_adjustment.md`
- `slam-core/src/triorb_slam_pipeline/src/gpu/cuda_bundle_adjustment_wrapper.cpp`
