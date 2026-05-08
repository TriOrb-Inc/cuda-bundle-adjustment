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

## 目標

- upstream 更新や参照比較時に、この file を導入している理由と利用位置を短時間で確認できる状態を保つ。
- project 側の差分が必要になった場合でも、変更理由を wrapper 側文書と合わせて追えるようにする。
- Schur complement 構造の host-side dedup と GPU-side pair enumeration の差を明示し、bad input があっても CUDA illegal address ではなく診断可能な失敗へ閉じる。

## 関連

- `slam-core/3rd/cuda-bundle-adjustment/src/cuda_bundle_adjustment.cpp`
- `slam-core/3rd/cuda-bundle-adjustment/src/cuda_bundle_adjustment.md`
- `slam-core/3rd/cuda-bundle-adjustment/src/cuda_linear_solver.md`
- `slam-core/3rd/cuda-bundle-adjustment/src/sparse_block_matrix.md`
