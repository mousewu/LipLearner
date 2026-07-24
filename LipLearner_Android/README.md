# LipLearner for Android

An Android (Kotlin) port of the iOS **LipLearner** app — customizable, on-device silent-speech
interaction. This is a functional re-implementation of `LipLearner_iOS`, built around the same
pretrained lip encoder.

## Architecture mapping (iOS → Android)

| iOS | Android |
|-----|---------|
| CoreML `LipEncoder.mlpackage` | ONNX Runtime Mobile + `lip_encoder.onnx` (`ml/LipEncoder.kt`) |
| Vision face / landmark detection | MediaPipe Face Landmarker (`vision/LipLandmarker.kt`) |
| CreateML `MLLogisticRegressionClassifier` (on-device training) | `ml/SoftmaxRegression.kt` |
| `commandCenterDict`, CSV/`.dat` persistence | `ml/CommandStore.kt` (JSON) |
| `SFSpeechRecognizer` (Voice2Lip) | `speech/SpeechRecognizerHelper.kt` |
| AVCaptureSession | CameraX (`camera/CameraController.kt`) |
| KWS / SSAD / recording loop in `CameraViewController` | `kws/LipRecognitionController.kt` |
| iOS Shortcuts execution | `exec/CommandExecutor.kt` (app launch / URI / broadcast to Tasker etc.) |
| UIKit `CameraViewController` | `ui/MainActivity.kt` |

## Prerequisites

- Android Studio (Giraffe+) or the command-line SDK, JDK 17, Android SDK 34.
- A device with Android 8.0+ (minSdk 26). Recommended: a recent mid/high-end phone — the encoder
  is a 3D-CNN + BiGRU and benefits from a fast CPU/NNAPI.

## Models (selectable at runtime)

The app supports **multiple interchangeable encoders**, chosen from **⋮ → Select model**. Each model
is an ONNX file in `app/src/main/assets/`; `ModelRegistry` lists them and only shows the ones whose
asset is present. All emit a 500-D embedding from 88×88 grayscale lip frames (per-model
normalization is baked into the ONNX graph). Learned commands are stored **per model** (different
models live in different embedding spaces), so switching models keeps each model's data separate.

| Model id | Backbone | LRW acc. | Size | Clip length |
|----------|----------|----------|------|-------------|
| `mpc001_resnet18_mstcn`   | ResNet18 + MS-TCN         | 88.9% | 145 MB | resampled to 29 |
| `mpc001_snv1x_dsmstcn3x`  | ShuffleNetV2 1× + DS-MSTCN| 85.3% | 37 MB  | resampled to 29 |
| `mpc001_snv1x_tcn1x`      | ShuffleNetV2 1× + TCN     | 82.7% | 15 MB  | resampled to 29 |
| `mpc001_snv05x_tcn1x`     | ShuffleNetV2 0.5× + TCN   | 79.9% | 12 MB  | resampled to 29 (lightest) |
| `liplearner` *(optional)* | ResNet18 + BiGRU (contrastive) | — | 120 MB | variable 10–128 |

The four **mpc001** models are already exported into `assets/` (from the public checkpoints via
`tools/export_mpc001_onnx.py`), so the app is fully functional out of the box.

### Adding the original LipLearner encoder (optional)

`lip_encoder.onnx` is not committed — generate it from the PyTorch weights:

1. Download `LipLearner_pretrained_model.pt` from the Google Drive link in the top-level `README.md`
   (must be done from a browser signed in to Google; the link is rate-limited for anonymous access).
2. Export and copy into assets:
   ```bash
   LipLearner_Android/scripts/build_model.sh /path/to/LipLearner_pretrained_model.pt
   ```
   It becomes selectable automatically once present.

> ⚠️ **License**: the mpc001 checkpoints are trained on LRW and are **research / non-commercial**
> only. Do not ship them in a commercial product.
>
> ⚠️ **Domain gap**: mpc001 models were trained on tightly dlib-aligned mouth crops. This app crops
> via MediaPipe (lip-centered, faceWidth×0.75). Because you register your *own* samples through the
> same crop, few-shot personalization still works, but absolute LRW-style accuracy will differ.
>
> **APK size**: bundling all four ONNX models makes the debug APK large (~330 MB). To slim it, delete
> unused `.onnx` files from `assets/` before building (the registry hides missing ones), or ask for
> INT8/fp16 quantization.

## Step 2 — build the APK

```bash
cd LipLearner_Android
./gradlew :app:assembleDebug
# output: app/build/outputs/apk/debug/app-debug.apk
```

## Step 3 — install & run

```bash
adb install -r app/build/outputs/apk/debug/app-debug.apk
```

Grant **Camera** and **Microphone** permissions on first launch.

## Usage (same model as the paper)

1. **Register** mode:
   - Record mode **Keyword**: capture a few activation-keyword samples.
   - Record mode **Non speaking**: capture a few mouth-closed / idle samples (used for EOS + SSAD).
   - Record mode **Command**: long-press the shutter, say a command aloud (Voice2Lip transcribes it
     to a label), confirm.
2. Toggle **KWS** for hands-free activation (needs keyword + non-speaking samples first).
3. **⋮ → Save and Train** to (re)train the classifiers and persist.
4. **Recognize** mode: confirm/correct predictions to add samples (active learning).
5. **Free use** mode: recognized commands are executed via `CommandExecutor`.

### Command execution mapping

Android has no Shortcuts. `CommandExecutor` resolves a command label via a saved mapping
(`app:<pkg>`, `url:<uri>`, `intent:<action>`) and otherwise broadcasts
`com.rkmtlab.liplearner.COMMAND` with a `command` extra — catchable by Tasker / MacroDroid /
Automate to run an arbitrary routine.

## Notes / differences from iOS

- SSAD gates KWS on `MOD >= 0.1` exactly like iOS (the "trick" not used in the paper's user study).
- The GIF-based recent-utterance review is not yet ported; recent free-use labels are kept in
  memory for the session.
- Effective analysis FPS depends on the device; CameraX drops frames under load
  (`STRATEGY_KEEP_ONLY_LATEST`).
