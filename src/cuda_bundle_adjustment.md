# cuda_bundle_adjustment.cpp

## 目的

`cuda_bundle_adjustment` は GPU 上の bundle adjustment 計算を提供する module です。同梱コードまたは参照実装の役割を、このリポジトリの文脈で素早く把握できるようにします。

## 対象範囲

- 対象 source: `slam-core/3rd/cuda-bundle-adjustment/src/cuda_bundle_adjustment.cpp`
- 判定条件: 915 行のため sidecar 文書を維持対象にする
- 主な定義: `CudaBlockSolver`、`ProfileItem`、`PLIndex`、`CudaBundleAdjustmentImpl`、`get_time_point`、`get_duration`、`ScalarCast`、`IntCast`
- 主な依存: `cuda_bundle_adjustment.h`、`algorithm`、`unordered_map`、`unordered_set`、`chrono`、`constants.h`

## 現状

### 主な構成
- `CudaBlockSolver` がこの module の主要な構成要素になっている
- `ProfileItem` がこの module の主要な構成要素になっている
- `PLIndex` がこの module の主要な構成要素になっている
- `CudaBundleAdjustmentImpl` がこの module の主要な構成要素になっている
- `get_time_point` がこの module の主要な構成要素になっている
- `get_duration` がこの module の主要な構成要素になっている

### 連携境界
- `cuda_bundle_adjustment.h` と連携しながら責務を完結させる
- `algorithm` と連携しながら責務を完結させる
- `unordered_map` と連携しながら責務を完結させる
- `unordered_set` と連携しながら責務を完結させる
- `chrono` と連携しながら責務を完結させる
- `constants.h` と連携しながら責務を完結させる

## 実装上の判断

- 同梱コードは upstream や参照実装としての責務を尊重し、この文書ではこのリポジトリから見た役割に絞って説明する。
- project 固有の判断は wrapper や利用側へ寄せ、この file 自体の変更理由を追いやすくする。
- 長大 file でも source 本体は read-only 前提で扱い、補足説明は sidecar 文書へ追加する。
- `TRIORB_CUDA_BA_TRACE=1` を付けると、Schur complement、CSR 変換、線形 solver 呼び出し、LM iteration の前後を `stderr` へ出して、Jetson 上の hang 箇所を段階別に追える。
- joint extrinsics solve (`TRIORB_OPTIMIZE_EXTRINSICS_JOINT=1`) では、`CudaBlockSolver` が host 側で edge ごとの direct `Hsc(body, ext)` slot (`edge2HscPE_`) を解決し、GPU 側の `HscDirect` へ `Jp^TΩJe` を積ませる。`Hpl` / `Hschur` だけでは direct pose-ext coupling を表現できず `factorize failed` に落ちるため、host/device 両方でこの追加配線を持つ。
- Schur multiply index 用の `d_HscMulBlockIds_` は、`Hsc_.nmulBlocks()` ではなく `HplBlockPos_` の landmark column ごとの row-slot pair 数 `sum(n * (n + 1) / 2)` から確保する。`Hsc_.nmulBlocks()` は unique Schur row pair 数であり、multi-camera / joint-ext edge で `Hpl` 側に duplicate slot が残る problem では kernel の実書き込み数より小さくなる。
- `TRIORB_CUDA_BA_TRACE=1` では `Hschur multiply slots: hsc_unique_pairs=..., hpl_pair_capacity=..., hpl_blocks=...` を出し、`Hsc` dedup 数と実際の `Hpl` pair 列挙上限を比較できるようにしている。
- `TRIORB_CUDA_BA_DETERMINISTIC_ACCUM=1` では、通常の body-only Local BA でも pose block `Hpp` / `bp` を fixed-point int64 mirror 経由で積算する。joint extrinsics 用に導入された `d_Hpp_int_ext_` / `d_bp_int_ext_` は名前を維持しているが、現在は全 pose slot を覆う full-size mirror として扱う。これにより body-only Local BA が `atomicAdd(double*)` に戻る経路を避ける。
- `CudaBundleAdjustmentImpl` の top-level 2D / 3D edge container は `std::vector` とし、wrapper が受け取った FFI edge 配列の追加順を保つ。以前の `std::unordered_set<Edge*>` は pointer hash の iteration order に依存し、同一 `VisualBaProblem` でも `computeErrors()` の edge reduction order と `initial_chi2` が run 間で変わる原因になっていた。重複挿入は別の `std::unordered_set` で O(1) に保ち、既存の connected-edge set は vertex 側の `unordered_set` に残す。

## 目標

- upstream 更新や参照比較時に、この file を導入している理由と利用位置を短時間で確認できる状態を保つ。
- project 側の差分が必要になった場合でも、変更理由を wrapper 側文書と合わせて追えるようにする。
- Schur complement 構造の host-side dedup と GPU-side pair enumeration の差を明示し、bad input があっても CUDA illegal address ではなく診断可能な失敗へ閉じる。

## 関連

- `slam-core/3rd/cuda-bundle-adjustment/src/cuda_bundle_adjustment.cpp`
- `slam-core/3rd/cuda-bundle-adjustment/src/cuda_bundle_adjustment.md`
- `slam-core/3rd/cuda-bundle-adjustment/src/cuda_linear_solver.md`
- `slam-core/3rd/cuda-bundle-adjustment/src/sparse_block_matrix.md`
