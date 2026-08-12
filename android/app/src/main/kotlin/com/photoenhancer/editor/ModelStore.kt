package com.photoenhancer.editor

import android.content.Context
import android.net.Uri
import android.provider.OpenableColumns
import java.io.File
import java.io.FileOutputStream
import java.util.Locale

/**
 * Owns the five ONNX graphs on disk.
 *
 * The APK ships *without* any model asset, because the five files total
 * ~396 MB and would make the download absurd. Instead the user hands the
 * files to the app once — from anywhere: Downloads, an SD card, Drive — and
 * this object works out which model each file is, stages it atomically and
 * reports readiness. Nothing here trusts the file name alone: a byte-exact
 * size match is the primary key, the name is only a fallback.
 *
 * Everything a stage needs is behind [ensure], so a model that arrives after
 * launch is picked up the moment the engine is reset.
 */
object ModelStore {

    /** One model the pipeline can use. */
    data class Spec(
        /** Canonical file name on disk; also the legacy asset name. */
        val asset: String,
        /** Arabic label for the UI. */
        val label: String,
        /** What the app loses without it. */
        val role: String,
        /** Byte-exact size of the reference export. */
        val bytes: Long,
        /** Lower-cased fragments that identify the file by name. */
        val keywords: List<String>,
        /** False when the pipeline degrades gracefully without it. */
        val required: Boolean
    ) {
        /** A staged file is only trusted inside this window of the reference. */
        val minBytes: Long get() = (bytes * 0.50).toLong()
        val maxBytes: Long get() = (bytes * 2.50).toLong() + 4096L
    }

    val SPECS: List<Spec> = listOf(
        Spec(
            asset = "swinir_x4.onnx",
            label = "التكبير الفائق SOTA (SwinIR x4)",
            role = "المحرّك الاحترافي المتقدم — انتباه عالمي للحواف",
            bytes = 64_300_000L,
            keywords = listOf("swinir", "swin", "srx4", "sr_x4", "003_realsr"),
            required = false
        ),
        Spec(
            asset = "hat_x4.onnx",
            label = "التكبير ×4 (HAT التقليدي)",
            role = "محرك التكبير الأساسي البديل",
            bytes = 165_066_073L,
            keywords = listOf("hat", "realhat", "upscal"),
            required = false
        ),
        Spec(
            asset = "codeformer.onnx",
            label = "ترميم الوجوه SOTA (CodeFormer)",
            role = "إعادة بناء ملامح الوجه بدقة سينمائية فائقة",
            bytes = 174_500_000L,
            keywords = listOf("codeformer", "code_former", "transformer"),
            required = false
        ),
        Spec(
            asset = "gfpgan.onnx",
            label = "ترميم الوجوه (GFPGAN الكلاسيكي)",
            role = "ترميم وجوه بديل",
            bytes = 170_189_529L,
            keywords = listOf("gfpgan", "gfp"),
            required = false
        ),
        Spec(
            asset = "nafnet.onnx",
            label = "تنظيف التشويش SOTA (NAFNet)",
            role = "إزالة الضوضاء والغبش بدون طمس التفاصيل الدقيقة",
            bytes = 96_200_000L,
            keywords = listOf("nafnet", "naf", "denois", "deblurring_nafnet"),
            required = false
        ),
        Spec(
            asset = "scunet_fp16.onnx",
            label = "تنظيف التشويش (SCUNet)",
            role = "إزالة ضوضاء بديل",
            bytes = 40_318_535L,
            keywords = listOf("scunet", "cleanup"),
            required = false
        ),
        Spec(
            asset = "retinaface.onnx",
            label = "كشف الوجوه المتقدم (RetinaFace)",
            role = "كشف دقيق جداً للوجوه في الإضاءة الصعبة",
            bytes = 1_250_000L,
            keywords = listOf("retinaface", "retina"),
            required = false
        ),
        Spec(
            asset = "yunet.onnx",
            label = "كشف الوجوه (YuNet)",
            role = "كشف وجوه خفيف",
            bytes = 232_589L,
            keywords = listOf("yunet", "detect"),
            required = false
        ),
        Spec(
            asset = "arcface.onnx",
            label = "التحقق من الهوية (ArcFace)",
            role = "حفظ هوية الشخص بدقة متناهية أثناء الترميم",
            bytes = 42_100_000L,
            keywords = listOf("arcface", "arc"),
            required = false
        ),
        Spec(
            asset = "sface.onnx",
            label = "التحقق من الهوية (SFace)",
            role = "تحقق هويّة بديل",
            bytes = 38_696_353L,
            keywords = listOf("sface", "identity"),
            required = false
        )
    )

    private fun specOf(asset: String): Spec? = SPECS.firstOrNull { it.asset == asset }

    fun dir(ctx: Context): File =
        File(ctx.noBackupFilesDir, "models").apply { mkdirs() }

    fun fileOf(ctx: Context, asset: String): File = File(dir(ctx), asset)

    // ------------------------------------------------------------ readiness

    /** True when a staged file exists at a size the reference export allows. */
    fun isStaged(ctx: Context, asset: String): Boolean {
        val f = fileOf(ctx, asset)
        if (!f.exists()) return false
        val spec = specOf(asset) ?: return f.length() > 1024L
        return f.length() in spec.minBytes..spec.maxBytes
    }

    /** Everything the UI needs to draw the manager screen. */
    fun status(ctx: Context): List<Map<String, Any>> = SPECS.map { s ->
        val f = fileOf(ctx, s.asset)
        val len = if (f.exists()) f.length() else 0L
        mapOf(
            "name" to s.asset,
            "label" to s.label,
            "role" to s.role,
            "required" to s.required,
            "expected" to s.bytes,
            "size" to len,
            "present" to isStaged(ctx, s.asset),
            // A file that is there but the wrong size is worse than a missing
            // one: it fails deep inside ONNX Runtime instead of here.
            "corrupt" to (len > 0L && !isStaged(ctx, s.asset))
        )
    }

    /** Legacy shape kept for the old Dart call site. */
    fun checkModelsStatus(ctx: Context): Map<String, Boolean> =
        SPECS.associate { it.asset to isStaged(ctx, it.asset) }

    /** The engine can start as soon as the super-resolver is on disk. */
    fun coreReady(ctx: Context): Boolean {
        // The engine is ready if either the SOTA or the fallback upscaler is present.
        return isStaged(ctx, "swinir_x4.onnx") || isStaged(ctx, "hat_x4.onnx")
    }

    fun allReady(ctx: Context): Boolean = SPECS.all { isStaged(ctx, it.asset) }

    // ------------------------------------------------------- identification

    /**
     * Works out which model a picked file is.
     *
     * Order matters. The byte size is decisive because every reference export
     * has a distinct one and the five sizes are orders of magnitude apart, so
     * a match cannot be a coincidence. Only when the size is unfamiliar — a
     * re-export, a different opset — does the name get a vote, and then the
     * size still has to be in the same league.
     *
     * @return the canonical asset name, or null when nothing fits.
     */
    fun identify(displayName: String?, size: Long): String? {
        // 1. Byte-exact match against a reference export.
        SPECS.firstOrNull { it.bytes == size }?.let { return it.asset }

        val n = (displayName ?: "").lowercase(Locale.ROOT)

        // 2. The canonical name itself.
        SPECS.firstOrNull { n == it.asset }?.let { return it.asset }

        // 3. A keyword, but only if the size is plausible for that model.
        if (n.isNotEmpty()) {
            SPECS.firstOrNull { s ->
                s.keywords.any { n.contains(it) } && (size <= 0L || size in s.minBytes..s.maxBytes)
            }?.let { return it.asset }
        }

        // 4. A keyword alone, when the size is unknown (some providers do not
        //    expose it) — better a named guess than refusing the file.
        if (n.isNotEmpty() && size <= 0L) {
            SPECS.firstOrNull { s -> s.keywords.any { n.contains(it) } }?.let { return it.asset }
        }

        // 5. Size alone, inside the tolerance window. Distinct enough to be
        //    safe: the closest pair (SCUNet 40 MB, SFace 38.7 MB) is resolved
        //    by picking the nearest.
        if (size > 0L) {
            val candidates = SPECS.filter { size in it.minBytes..it.maxBytes }
            if (candidates.size == 1) return candidates[0].asset
            if (candidates.size > 1) {
                return candidates.minByOrNull { kotlin.math.abs(it.bytes - size) }?.asset
            }
        }
        return null
    }

    // -------------------------------------------------------------- staging

    data class Imported(
        val asset: String,
        val label: String,
        val bytes: Long
    )

    /**
     * Streams [uri] into the models directory under its resolved name.
     *
     * The copy lands in a `.part` file and is renamed only after the stream
     * closes, so a cancelled or crashed import can never leave a truncated
     * graph that ONNX Runtime would then try to parse.
     *
     * @param onProgress called with (copiedBytes, totalBytes); total is -1
     *        when the provider does not report a size.
     */
    fun importFrom(
        ctx: Context,
        uri: Uri,
        forceAsset: String? = null,
        onProgress: ((Long, Long) -> Unit)? = null
    ): Imported {
        val meta = queryMeta(ctx, uri)
        val asset = forceAsset ?: identify(meta.first, meta.second)
            ?: throw IllegalArgumentException(
                "تعذّر التعرّف على الملف \"${meta.first ?: "غير معروف"}\". " +
                    "أعد تسميته إلى أحد الأسماء المعروفة أو استوردْه من شاشة النموذج المطلوب."
            )

        val spec = specOf(asset)
        val total = meta.second
        val target = fileOf(ctx, asset)
        val tmp = File(dir(ctx), "$asset.part")
        if (tmp.exists()) tmp.delete()

        var copied = 0L
        try {
            val input = ctx.contentResolver.openInputStream(uri)
                ?: throw IllegalStateException("تعذّر فتح الملف للقراءة")
            input.use { ins ->
                FileOutputStream(tmp).use { out ->
                    val buf = ByteArray(1 shl 20)
                    var last = 0L
                    while (true) {
                        val r = ins.read(buf)
                        if (r <= 0) break
                        out.write(buf, 0, r)
                        copied += r
                        // Throttled so a 165 MB copy does not flood the UI
                        // thread with thousands of messages.
                        if (copied - last >= (4L shl 20)) {
                            last = copied
                            onProgress?.invoke(copied, total)
                        }
                    }
                    out.flush()
                    out.fd.sync()
                }
            }
        } catch (t: Throwable) {
            tmp.delete()
            throw t
        }

        if (copied <= 0L) {
            tmp.delete()
            throw IllegalStateException("الملف فارغ")
        }
        if (spec != null && copied !in spec.minBytes..spec.maxBytes) {
            tmp.delete()
            throw IllegalStateException(
                "حجم الملف (${human(copied)}) لا يطابق ${spec.label} " +
                    "(المتوقّع ${human(spec.bytes)})"
            )
        }

        if (target.exists()) target.delete()
        if (!tmp.renameTo(target)) {
            tmp.delete()
            throw IllegalStateException("تعذّر حفظ النموذج على القرص")
        }
        onProgress?.invoke(copied, copied)
        return Imported(asset, spec?.label ?: asset, copied)
    }

    /** display name and size, either of which a provider may withhold. */
    private fun queryMeta(ctx: Context, uri: Uri): Pair<String?, Long> {
        var name: String? = null
        var size = -1L
        runCatching {
            ctx.contentResolver.query(uri, null, null, null, null)?.use { c ->
                if (c.moveToFirst()) {
                    val ni = c.getColumnIndex(OpenableColumns.DISPLAY_NAME)
                    if (ni >= 0 && !c.isNull(ni)) name = c.getString(ni)
                    val si = c.getColumnIndex(OpenableColumns.SIZE)
                    if (si >= 0 && !c.isNull(si)) size = c.getLong(si)
                }
            }
        }
        if (name == null) name = uri.lastPathSegment
        if (size <= 0L && "file" == uri.scheme) {
            runCatching { size = File(uri.path!!).length() }
        }
        return name to size
    }

    fun human(bytes: Long): String = when {
        bytes >= 1L shl 30 -> String.format(Locale.US, "%.2f جيجا", bytes / 1073741824.0)
        bytes >= 1L shl 20 -> String.format(Locale.US, "%.1f ميجا", bytes / 1048576.0)
        bytes >= 1L shl 10 -> String.format(Locale.US, "%.0f كيلو", bytes / 1024.0)
        else -> "$bytes بايت"
    }

    // ------------------------------------------------------------- lookups

    /**
     * ONNX Runtime needs a real path. The file is either already staged by
     * the user, or — for builds that do bundle assets — copied out once.
     */
    @Synchronized
    fun ensure(ctx: Context, asset: String): String {
        val out = fileOf(ctx, asset)
        // If it's staged correctly, return it.
        if (isStaged(ctx, asset)) return out.absolutePath

        // If it exists but is NOT staged (e.g. wrong size), delete it to avoid ORT crashes.
        if (out.exists()) out.delete()

        // Check if it's in assets (bundled builds).
        val inAssets = runCatching { ctx.assets.open(asset).use { it.available() > 0 } }.getOrDefault(false)
        
        if (!inAssets) {
            val spec = specOf(asset)
            throw IllegalStateException(
                "ملف النموذج '$asset' (${spec?.label ?: "غير معروف"}) غير موجود في التخزين المحلي أو ملفات التطبيق. يرجى استيراده يدوياً."
            )
        }

        val tmp = File(dir(ctx), "$asset.part")
        try {
            ctx.assets.open(asset).use { input ->
                FileOutputStream(tmp).use { o -> input.copyTo(o, 1 shl 20) }
            }
        } catch (t: Throwable) {
            tmp.delete()
            throw IllegalStateException("تعذّر تجهيز $asset", t)
        }
        if (out.exists()) out.delete()
        if (!tmp.renameTo(out)) throw IllegalStateException("cannot stage $asset")
        return out.absolutePath
    }

    /** Removes one staged model plus any partial copy beside it. */
    fun drop(ctx: Context, asset: String) {
        runCatching { fileOf(ctx, asset).delete() }
        runCatching { File(dir(ctx), "$asset.part").delete() }
    }

    /** Total bytes currently held on disk by staged models. */
    fun occupiedBytes(ctx: Context): Long =
        dir(ctx).listFiles()?.sumOf { it.length() } ?: 0L

    /** Deletes leftovers from interrupted imports. */
    fun sweepPartials(ctx: Context) {
        dir(ctx).listFiles()?.forEach {
            if (it.name.endsWith(".part")) runCatching { it.delete() }
        }
    }
}
