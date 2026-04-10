# cuda_linear_solver.cpp / cuda_linear_solver.h

## 目的

`cuda_linear_solver` は GPU 線形 solver の土台を提供する module です。同梱コードまたは参照実装の役割を、このリポジトリの文脈で素早く把握できるようにします。

## 対象範囲

- 対象 source: `slam-core/3rd/cuda-bundle-adjustment/src/cuda_linear_solver.cpp`、`slam-core/3rd/cuda-bundle-adjustment/src/cuda_linear_solver.h`
- 判定条件: 477 行のため sidecar 文書を維持対象にする
- 主な定義: `CusparseHandle`、`CusolverHandle`、`CusparseMatDescriptor`、`SparseSquareMatrixCSR`、`SparseCholesky`、`CuSparseCholeskySolver`、`SparseLinearSolverImpl`、`SparseLinearSolver`
- 主な依存: `Eigen/Core`、`Eigen/Sparse`、`cuda_linear_solver.h`、`iostream`、`vector`、`cuda_runtime.h`

## 現状

### 主な構成
- `CusparseHandle` がこの module の主要な構成要素になっている
- `CusolverHandle` がこの module の主要な構成要素になっている
- `CusparseMatDescriptor` がこの module の主要な構成要素になっている
- `SparseSquareMatrixCSR` がこの module の主要な構成要素になっている
- `SparseCholesky` がこの module の主要な構成要素になっている
- `CuSparseCholeskySolver` がこの module の主要な構成要素になっている

### 連携境界
- `Eigen/Core` と連携しながら責務を完結させる
- `Eigen/Sparse` と連携しながら責務を完結させる
- `cuda_linear_solver.h` と連携しながら責務を完結させる
- `iostream` と連携しながら責務を完結させる
- `vector` と連携しながら責務を完結させる
- `cuda_runtime.h` と連携しながら責務を完結させる

## 実装上の判断

- 同梱コードは upstream や参照実装としての責務を尊重し、この文書ではこのリポジトリから見た役割に絞って説明する。
- project 固有の判断は wrapper や利用側へ寄せ、この file 自体の変更理由を追いやすくする。
- 長大 file でも source 本体は read-only 前提で扱い、補足説明は sidecar 文書へ追加する。

## 目標

- upstream 更新や参照比較時に、この file を導入している理由と利用位置を短時間で確認できる状態を保つ。
- project 側の差分が必要になった場合でも、変更理由を wrapper 側文書と合わせて追えるようにする。

## 関連

- `slam-core/3rd/cuda-bundle-adjustment/src/cuda_linear_solver.cpp`
- `slam-core/3rd/cuda-bundle-adjustment/src/cuda_linear_solver.h`
- `slam-core/3rd/cuda-bundle-adjustment/src/cuda_linear_solver.md`
- `slam-core/3rd/cuda-bundle-adjustment/src/cuda_bundle_adjustment.md`
- `slam-core/3rd/cuda-bundle-adjustment/src/sparse_block_matrix.md`
