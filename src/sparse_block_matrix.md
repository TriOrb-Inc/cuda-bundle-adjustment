# sparse_block_matrix.cpp / sparse_block_matrix.h

## 目的

`sparse_block_matrix` は 疎 block matrix 表現と演算を提供する module です。同梱コードまたは参照実装の役割を、このリポジトリの文脈で素早く把握できるようにします。

## 対象範囲

- 対象 source: `slam-core/3rd/cuda-bundle-adjustment/src/sparse_block_matrix.cpp`、`slam-core/3rd/cuda-bundle-adjustment/src/sparse_block_matrix.h`
- 判定条件: 335 行のため sidecar 文書を維持対象にする
- 主な定義: `BlockPos`、`SparseBlockMatrix`、`HplBlockPos`、`HplSparseBlockMatrix`、`HschurSparseBlockMatrix`、`resize`、`resizeNonzeros`、`outerIndices`
- 主な依存: `Eigen/Core`、`sparse_block_matrix.h`、`algorithm`、`vector`、`cuda_bundle_adjustment_types.h`、`constants.h`

## 現状

### 主な構成
- `BlockPos` がこの module の主要な構成要素になっている
- `SparseBlockMatrix` がこの module の主要な構成要素になっている
- `HplBlockPos` がこの module の主要な構成要素になっている
- `HplSparseBlockMatrix` がこの module の主要な構成要素になっている
- `HschurSparseBlockMatrix` がこの module の主要な構成要素になっている
- `resize` がこの module の主要な構成要素になっている
- `HschurSparseBlockMatrix::constructFromVertices()` は、landmark ごとの active row 集合を `Hpl` と同じ unique な `iP` 集合として扱う。joint extrinsics では同じ extrinsics vertex が同じ landmark を複数 edge で参照するため、edge incidence をそのまま数えると `nmultiplies_` が過大になり、Schur metadata と `Hpl` の row-pair 列挙がずれる。

### 連携境界
- `Eigen/Core` と連携しながら責務を完結させる
- `sparse_block_matrix.h` と連携しながら責務を完結させる
- `algorithm` と連携しながら責務を完結させる
- `vector` と連携しながら責務を完結させる
- `cuda_bundle_adjustment_types.h` と連携しながら責務を完結させる
- `constants.h` と連携しながら責務を完結させる

## 実装上の判断

- 同梱コードは upstream や参照実装としての責務を尊重し、この文書ではこのリポジトリから見た役割に絞って説明する。
- project 固有の判断は wrapper や利用側へ寄せ、この file 自体の変更理由を追いやすくする。
- 長大 file でも source 本体は read-only 前提で扱い、補足説明は sidecar 文書へ追加する。

## 目標

- upstream 更新や参照比較時に、この file を導入している理由と利用位置を短時間で確認できる状態を保つ。
- project 側の差分が必要になった場合でも、変更理由を wrapper 側文書と合わせて追えるようにする。

## 関連

- `slam-core/3rd/cuda-bundle-adjustment/src/sparse_block_matrix.cpp`
- `slam-core/3rd/cuda-bundle-adjustment/src/sparse_block_matrix.h`
- `slam-core/3rd/cuda-bundle-adjustment/src/sparse_block_matrix.md`
- `slam-core/3rd/cuda-bundle-adjustment/src/cuda_bundle_adjustment.md`
- `slam-core/3rd/cuda-bundle-adjustment/src/cuda_linear_solver.md`
