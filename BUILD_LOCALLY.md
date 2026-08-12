# البناء محليًا

## المتطلبات

- Flutter **3.35.4** · Dart **3.9.2** (مثبّتة تمامًا — لا ترقّها)
- JDK **17**
- Android SDK: compileSdk **36** · buildTools **35.0.0** · minSdk **28**

## النماذج غير مضمّنة في الأرشيف

الأرشيف يحتوي الكود فقط (188 كيلوبايت). النماذج الخمسة حجمها **396 ميجابايت**
ولذلك استُثنيت. ضعها في `android/app/src/main/assets/` بهذه الأحجام بالضبط:

| الملف | الحجم بالبايت | المصدر |
|---|---|---|
| `hat_x4.onnx` | 165,066,073 | Real_HAT_GAN_SRx4_sharper |
| `gfpgan.onnx` | 170,189,529 | GFPGANv1.4 |
| `scunet_fp16.onnx` | 40,318,535 | SCUNet color_real_psnr |
| `sface.onnx` | 38,696,353 | OpenCV Zoo SFace |
| `yunet.onnx` | 232,589 | OpenCV Zoo YuNet |

بدون هذه الملفات سيُبنى التطبيق بنجاح لكن `ModelStore.ensure` سيفشل وقت
التشغيل، فتظهر الواجهة في وضع «معاينة».

## التوقيع

`android/key.properties` و`release-key.jks` مستثنيان أيضًا (مفاتيح خاصة).
لبناء release وقّع بمفتاحك:

```properties
# android/key.properties
storePassword=...
keyPassword=...
keyAlias=...
storeFile=release-key.jks
```

أو ابنِ debug مباشرة بلا توقيع.

## الأوامر

```bash
flutter pub get
flutter analyze                 # يجب أن يكون نظيفًا
flutter build apk --release --target-platform android-arm64
```

الحجم المتوقّع: **‏437.6 ميجابايت** · `arm64-v8a` فقط.

## خريطة الكود

الأنبوب ست مراحل، منسّقها `Pipeline.kt`:

| # | المرحلة | الملف | النموذج |
|---|---|---|---|
| 1 | تحليل التلف | `QualityAnalyzer.kt` | YuNet (وجوه) |
| 2 | تنظيف | `CleanupStage.kt` | SCUNet |
| 3 | تكبير ×4 | `SrEngine.kt` | HAT |
| 4 | ترميم الوجوه | `FaceStage.kt` | GFPGAN |
| 5 | دمج التفاصيل | `DetailFusion.kt` | — |
| 6 | بوابة الجودة | `QualityGate.kt` | SFace |

مساران مشتركان: `Accelerator.kt` (سلّم المسرّعات لكل النماذج) و
`ThermalGovernor.kt` (تهدئة حرارية) و`ImageIo.kt` (فك/ترميز) و
`ModelStore.kt` (نسخ الأصول إلى القرص).
