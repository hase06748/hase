package com.photoenhancer.editor

import ai.onnxruntime.OrtEnvironment
import android.content.Context
import android.graphics.Bitmap

/**
 * The six-stage restoration pipeline, in order:
 *
 *   1. Analyse   - QualityAnalyzer: noise, JPEG blocking, blur type, faces.
 *   2. Clean     - CleanupStage: denoise / deblock (SCUNet) then deblur.
 *   3. Upscale   - SrEngine: Real_HAT_GAN_SRx4_sharper, x4.
 *   4. Regions   - FaceStage: GFPGAN inside the face ROI only, skin protected.
 *   5. Fuse      - DetailFusion: local contrast + edge-gated sharpening.
 *   6. Gate      - QualityGate: per-region regression + face identity check.
 *
 * Stages 2, 4 and 6 are conditional: they only run when stage 1 says they are
 * needed, or when the user turns them off. That is deliberate — every stage
 * that touches pixels is also a stage that can spoil them, so none of them
 * runs "just in case".
 *
 * All heavy models are loaded lazily and released as soon as their stage is
 * finished, because holding HAT (165 MB), GFPGAN (163 MB) and SCUNet (40 MB)
 * resident at the same time would be reckless even on a 16 GB phone.
 */
object Pipeline {

    /** What the caller wants; all default to the highest-quality behaviour. */
    data class Options(
        val maxPixels: Long,
        val sharpen: Float,
        val cleanup: Boolean = true,
        val faceRestore: Boolean = true,
        val faceStrength: Float = 0.8f,
        val qualityGate: Boolean = true,
        val protectSkin: Boolean = true,
        val protectSky: Boolean = true
    )

    /** Coarse phase identifier, surfaced to the UI. */
    enum class Phase(val label: String) {
        ANALYZE("تحليل الصورة"),
        CLEANUP("تنظيف التشويش"),
        UPSCALE("تكبير ×4"),
        FACES("ترميم الوجوه"),
        FUSION("دمج التفاصيل"),
        GATE("فحص الجودة"),
        DONE("تم")
    }

    /** One progress report; [done]/[total] are within the current phase. */
    data class Progress(
        val phase: Phase,
        val done: Int,
        val total: Int,
        val tileMs: Long,
        val thermal: String,
        val headroomPercent: Int,
        val throttling: Boolean,
        val pausedMs: Long,
        val note: String
    )

    /** Everything worth telling the user once the run finishes. */
    data class Outcome(
        val bitmap: Bitmap,
        val analysis: QualityAnalyzer.Report,
        val cleanupApplied: Boolean,
        val facesRestored: Int,
        val verdict: QualityGate.Verdict?,
        val totalMs: Long
    ) {
        fun summary(): String {
            val parts = ArrayList<String>(4)
            parts.add(analysis.summary())
            if (cleanupApplied) parts.add("نُظّفت")
            if (facesRestored > 0) parts.add("$facesRestored وجه مُرمّم")
            verdict?.let { parts.add(it.summary()) }
            return parts.joinToString(" · ")
        }
    }

    /**
     * Runs the whole pipeline. Returns null when cancelled.
     *
     * @param source the working bitmap; every stage operates on this one.
     * @param native the untouched decode when [source] is a downscale of it.
     *               Stage 1 needs it because noise and JPEG blocking do not
     *               survive resampling; pass null when they are the same.
     * @param threads how many cores the neural stages may use.
     */
    fun run(
        ctx: Context,
        source: Bitmap,
        options: Options,
        threads: Int,
        native: Bitmap? = null,
        governor: ThermalGovernor?,
        cancelled: () -> Boolean,
        onProgress: (Progress) -> Unit
    ): Outcome? {
        val started = System.currentTimeMillis()
        SrEngine.clearCancel()
        val env = OrtEnvironment.getEnvironment()

        fun report(
            phase: Phase,
            done: Int,
            total: Int,
            tileMs: Long = 0L,
            note: String = ""
        ) {
            onProgress(
                Progress(
                    phase = phase,
                    done = done,
                    total = total,
                    tileMs = tileMs,
                    thermal = (governor?.level() ?: ThermalGovernor.Level.NORMAL).label,
                    headroomPercent = governor?.headroomPercent() ?: -1,
                    throttling = governor?.isThrottling() ?: false,
                    pausedMs = governor?.pausedMillis() ?: 0L,
                    note = note
                )
            )
        }

        // ---------------------------------------------------- 1. analyse
        report(Phase.ANALYZE, 0, 1)
        val detector = FaceDetector.create(ctx, env)
        val analysis = try {
            QualityAnalyzer.analyze(source, detector, native)
        } finally {
            detector?.close()
        }
        report(Phase.ANALYZE, 1, 1, analysis.analysisMs, analysis.summary())
        if (cancelled()) return null

        // ---------------------------------------------------- 2. clean
        var working = source
        var cleanupApplied = false
        val wantsCleanup = options.cleanup &&
            (analysis.needsDenoise || analysis.needsDeblock || analysis.needsDeblur)

        if (wantsCleanup) {
            val stage = CleanupStage.create(ctx, env, threads)
            if (stage != null) {
                try {
                    val total = stage.tileCount(source.width, source.height)
                    var n = 0
                    report(Phase.CLEANUP, 0, total)
                    val cleaned = stage.apply(source, analysis, governor, cancelled) { ms ->
                        n++
                        report(Phase.CLEANUP, n, total, ms)
                    }
                    if (cleaned !== source) {
                        working = cleaned
                        cleanupApplied = true
                    }
                } finally {
                    stage.close()
                }
            }
        }
        if (cancelled()) {
            if (working !== source) working.recycle()
            return null
        }

        // ---------------------------------------------------- 3. upscale
        val upscaled: Bitmap? = try {
            report(Phase.UPSCALE, 0, SrEngine.tileCount(working.width, working.height))
            SrEngine.upscale(working, governor) { tick ->
                onProgress(
                    Progress(
                        phase = Phase.UPSCALE,
                        done = tick.done,
                        total = tick.total,
                        tileMs = tick.tileMs,
                        thermal = tick.thermal,
                        headroomPercent = tick.headroomPercent,
                        throttling = tick.throttling,
                        pausedMs = tick.pausedMs,
                        note = ""
                    )
                )
            }
        } catch (t: Throwable) {
            if (working !== source) working.recycle()
            throw t
        }
        // NOTE: `working` deliberately stays alive past this point. When
        // cleanup ran, it — not `source` — is what the super-resolver actually
        // saw, and stage 6 must judge the output against that same reference.
        // Measuring against the noisy original made the gate revert cells for
        // carrying *less* noise energy than the input, which is not a defect.
        if (upscaled == null) {
            if (working !== source) working.recycle()
            return null
        }

        // ---------------------------------------------------- 4. regions
        var facesRestored = 0
        if (options.faceRestore && analysis.hasFaces && !cancelled()) {
            val stage = FaceStage.create(ctx, env, threads)
            if (stage != null) {
                try {
                    report(Phase.FACES, 0, analysis.faces.size)
                    facesRestored = stage.restore(
                        target = upscaled,
                        faces = analysis.faces,
                        scale = SrEngine.SCALE,
                        strength = options.faceStrength,
                        cancelled = cancelled
                    ) { n, ms ->
                        report(Phase.FACES, n, analysis.faces.size, ms)
                        governor?.poll()
                        governor?.coolDown(ms) { cancelled() }
                    }
                } finally {
                    stage.close()
                }
            }
        }
        if (cancelled()) {
            upscaled.recycle()
            if (working !== source) working.recycle()
            return null
        }

        // ---------------------------------------------------- 5. fusion
        if (options.sharpen > 0.01f) {
            report(Phase.FUSION, 0, 1)
            DetailFusion.apply(
                upscaled,
                options.sharpen,
                protectSkin = options.protectSkin,
                protectSky = options.protectSky
            )
            report(Phase.FUSION, 1, 1)
        }
        if (cancelled()) {
            upscaled.recycle()
            if (working !== source) working.recycle()
            return null
        }

        // ---------------------------------------------------- 6. gate
        var verdict: QualityGate.Verdict? = null
        if (options.qualityGate) {
            report(Phase.GATE, 0, 1)
            val gate = QualityGate.create(ctx, env)
            try {
                verdict = gate.evaluate(
                    source = working,
                    result = upscaled,
                    faces = analysis.faces,
                    scale = SrEngine.SCALE,
                    cancelled = cancelled
                )
            } finally {
                gate.close()
            }
            report(Phase.GATE, 1, 1, 0L, verdict?.summary() ?: "")
        }

        if (working !== source) working.recycle()

        report(Phase.DONE, 1, 1)
        return Outcome(
            bitmap = upscaled,
            analysis = analysis,
            cleanupApplied = cleanupApplied,
            facesRestored = facesRestored,
            verdict = verdict,
            totalMs = System.currentTimeMillis() - started
        )
    }
}
