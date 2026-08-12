package com.photoenhancer.editor

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Paint
import android.graphics.Rect
import java.io.Closeable
import java.nio.FloatBuffer
import kotlin.math.abs
import kotlin.math.max
import kotlin.math.min
import kotlin.math.roundToInt
import kotlin.math.sqrt

/**
 * Stage 6: the safety net.
 *
 * Every earlier stage can, in principle, make things worse: a denoiser can
 * erase fine texture, a generative face model can change who somebody looks
 * like, adaptive sharpening can ring. This stage asks two blunt questions and
 * acts on the answers.
 *
 *  1. Is any *region* worse than the original? The image is divided into a
 *     grid and each cell is compared for detail energy and for structural
 *     agreement with the (upscaled) source. A cell that lost detail or drifted
 *     structurally is faded back towards a clean bicubic upscale of the
 *     original, in proportion to how bad it is.
 *
 *  2. Did a face change beyond an acceptable threshold? SFace produces a
 *     128-dimensional identity embedding; we embed the same face before and
 *     after and take the cosine similarity. Below the threshold the face is
 *     reverted or partially blended, because a beautiful face that is not the
 *     right person is a failure, not an enhancement.
 *
 * Reverting is done as a blend rather than a hard swap, so a borderline cell
 * degrades gracefully instead of producing a visible patch.
 */
class QualityGate private constructor(
    private val env: OrtEnvironment,
    private val session: OrtSession?
) : Closeable {

    companion object {
        private const val ASSET = "sface.onnx"
        private const val SIDE = 112

        /** Cosine similarity below this means the identity drifted too far. */
        private const val IDENTITY_MIN = 0.55f

        /** Grid cell size in output pixels for the regression check. */
        private const val CELL = 128

        /**
         * The identity model is optional: if it cannot load, the region check
         * still runs, which is the more important of the two.
         */
        fun create(ctx: Context, env: OrtEnvironment): QualityGate {
            val s = try {
                val path = ModelStore.ensure(ctx, ASSET)
                // SFace produces an embedding compared by cosine distance
                // against a 0.55 threshold; fp16 noise is far below that
                // margin, so the relaxed rung is allowed.
                Accelerator.create(
                    env = env,
                    modelPath = path,
                    label = "هوية",
                    threads = 2,
                    precision = Accelerator.Precision.RELAXED
                )
            } catch (t: Throwable) {
                null
            }
            return QualityGate(env, s)
        }
    }

    /**
     * @param cellsChecked   how many grid cells were evaluated.
     * @param cellsReverted  how many were blended back towards the source.
     * @param facesChecked   how many faces were identity-tested.
     * @param facesReverted  how many faces failed and were pulled back.
     * @param worstSimilarity lowest cosine similarity seen, or -1.
     */
    data class Verdict(
        val cellsChecked: Int,
        val cellsReverted: Int,
        val facesChecked: Int,
        val facesReverted: Int,
        val worstSimilarity: Float
    ) {
        fun summary(): String = when {
            cellsReverted == 0 && facesReverted == 0 -> "اجتاز فحص الجودة"
            facesReverted > 0 -> "أُعيد $facesReverted وجه · $cellsReverted منطقة"
            else -> "أُعيدت $cellsReverted منطقة"
        }
    }

    /**
     * Runs both checks and repairs [result] in place.
     *
     * @param source the original (pre-pipeline) bitmap.
     * @param result the finished pipeline output; modified in place.
     * @param faces  face boxes in source coordinates.
     * @param scale  result/source scale factor.
     */
    fun evaluate(
        source: Bitmap,
        result: Bitmap,
        faces: List<FaceDetector.Face>,
        scale: Int,
        cancelled: () -> Boolean
    ): Verdict {
        var checked = 0
        var reverted = 0

        // --- 1. per-region regression -------------------------------------
        val w = result.width
        val h = result.height
        val cols = max(1, (w + CELL - 1) / CELL)
        val rows = max(1, (h + CELL - 1) / CELL)

        val refCell = IntArray(CELL * CELL)
        val outCell = IntArray(CELL * CELL)

        for (cy in 0 until rows) {
            if (cancelled()) break
            for (cx in 0 until cols) {
                val x0 = cx * CELL
                val y0 = cy * CELL
                val cw = min(CELL, w - x0)
                val ch = min(CELL, h - y0)
                if (cw < 16 || ch < 16) continue

                // Bicubic reference for this cell, straight from the source.
                val sx0 = x0 / scale
                val sy0 = y0 / scale
                val sw = max(2, cw / scale)
                val sh = max(2, ch / scale)
                if (sx0 + sw > source.width || sy0 + sh > source.height) continue

                val ref = Bitmap.createBitmap(cw, ch, Bitmap.Config.ARGB_8888)
                try {
                    Canvas(ref).drawBitmap(
                        source,
                        Rect(sx0, sy0, sx0 + sw, sy0 + sh),
                        Rect(0, 0, cw, ch),
                        Paint(Paint.FILTER_BITMAP_FLAG)
                    )
                    val n = cw * ch
                    ref.getPixels(refCell, 0, cw, 0, 0, cw, ch)
                    result.getPixels(outCell, 0, cw, x0, y0, cw, ch)
                    checked++

                    val dRef = detailEnergy(refCell, cw, ch)
                    val dOut = detailEnergy(outCell, cw, ch)
                    val corr = correlation(refCell, outCell, n)

                    // A cell fails if it lost detail relative to a plain
                    // upscale, or if it no longer resembles the source at all.
                    var penalty = 0f
                    if (dRef > 1.5f && dOut < dRef * 0.88f) {
                        penalty = max(penalty, ((dRef * 0.88f - dOut) / (dRef * 0.88f)).coerceIn(0f, 1f))
                    }
                    if (corr < 0.86f) {
                        penalty = max(penalty, ((0.86f - corr) / 0.36f).coerceIn(0f, 1f))
                    }
                    // Cap the pull-back: never discard the SR result entirely.
                    val wgt = min(0.7f, penalty)
                    if (wgt > 0.04f) {
                        for (i in 0 until n) {
                            val a = outCell[i]
                            val b = refCell[i]
                            outCell[i] = (0xFF shl 24) or
                                (mix((a shr 16) and 0xFF, (b shr 16) and 0xFF, wgt) shl 16) or
                                (mix((a shr 8) and 0xFF, (b shr 8) and 0xFF, wgt) shl 8) or
                                mix(a and 0xFF, b and 0xFF, wgt)
                        }
                        result.setPixels(outCell, 0, cw, x0, y0, cw, ch)
                        reverted++
                    }
                } finally {
                    ref.recycle()
                }
            }
        }

        // --- 2. face identity ---------------------------------------------
        var facesChecked = 0
        var facesReverted = 0
        var worst = -1f

        if (session != null && faces.isNotEmpty()) {
            for (face in faces) {
                if (cancelled()) break
                val b = face.box
                if (b.width() < 24 || b.height() < 24) continue

                val before = embed(source, b, 1)
                val after = embed(result, b, scale)
                if (before == null || after == null) continue
                facesChecked++
                val sim = cosine(before, after)
                if (worst < 0f || sim < worst) worst = sim

                if (sim < IDENTITY_MIN) {
                    // Pull the face back towards a plain upscale, proportional
                    // to how far the identity drifted.
                    val wgt = min(0.85f, ((IDENTITY_MIN - sim) / IDENTITY_MIN).coerceIn(0f, 1f))
                    revertRegion(source, result, b, scale, wgt)
                    facesReverted++
                }
            }
        }

        return Verdict(checked, reverted, facesChecked, facesReverted, worst)
    }

    // ------------------------------------------------------------ embedding

    /** Crops [box] (in [scale] units), resizes to 112 and returns the 128-D vector. */
    private fun embed(bmp: Bitmap, box: Rect, scale: Int): FloatArray? = try {
        val x0 = (box.left * scale).coerceIn(0, bmp.width - 2)
        val y0 = (box.top * scale).coerceIn(0, bmp.height - 2)
        val x1 = (box.right * scale).coerceIn(x0 + 2, bmp.width)
        val y1 = (box.bottom * scale).coerceIn(y0 + 2, bmp.height)

        val square = Bitmap.createBitmap(SIDE, SIDE, Bitmap.Config.ARGB_8888)
        try {
            Canvas(square).drawBitmap(
                bmp, Rect(x0, y0, x1, y1), Rect(0, 0, SIDE, SIDE),
                Paint(Paint.FILTER_BITMAP_FLAG)
            )
            val buf = FloatBuffer.allocate(3 * SIDE * SIDE)
            val arr = buf.array()
            val plane = SIDE * SIDE
            val row = IntArray(SIDE)
            for (y in 0 until SIDE) {
                square.getPixels(row, 0, SIDE, 0, y, SIDE, 1)
                val base = y * SIDE
                for (x in 0 until SIDE) {
                    val c = row[x]
                    // SFace is an OpenCV Zoo model: BGR, raw 0..255.
                    arr[base + x] = (c and 0xFF).toFloat()
                    arr[plane + base + x] = ((c shr 8) and 0xFF).toFloat()
                    arr[2 * plane + base + x] = ((c shr 16) and 0xFF).toFloat()
                }
            }
            buf.rewind()
            val s = session ?: return null
            val inName = s.inputNames.iterator().next()
            var vec: FloatArray? = null
            OnnxTensor.createTensor(env, buf, longArrayOf(1, 3, SIDE.toLong(), SIDE.toLong()))
                .use { tensor ->
                    s.run(mapOf(inName to tensor)).use { res ->
                        val fb = (res[0] as OnnxTensor).floatBuffer
                        val v = FloatArray(fb.capacity())
                        fb.get(v)
                        if (v.isNotEmpty() && !v[0].isNaN()) vec = v
                    }
                }
            vec
        } finally {
            square.recycle()
        }
    } catch (t: Throwable) {
        null
    }

    private fun cosine(a: FloatArray, b: FloatArray): Float {
        val n = min(a.size, b.size)
        var dot = 0.0
        var na = 0.0
        var nb = 0.0
        for (i in 0 until n) {
            dot += a[i] * b[i]
            na += a[i] * a[i]
            nb += b[i] * b[i]
        }
        if (na <= 1e-9 || nb <= 1e-9) return 0f
        return (dot / (sqrt(na) * sqrt(nb))).toFloat()
    }

    // --------------------------------------------------------------- helpers

    /** Blends the face region back towards a bicubic upscale of the source. */
    private fun revertRegion(
        source: Bitmap,
        result: Bitmap,
        box: Rect,
        scale: Int,
        weight: Float
    ) {
        val x0 = (box.left * scale).coerceIn(0, result.width - 2)
        val y0 = (box.top * scale).coerceIn(0, result.height - 2)
        val x1 = (box.right * scale).coerceIn(x0 + 2, result.width)
        val y1 = (box.bottom * scale).coerceIn(y0 + 2, result.height)
        val rw = x1 - x0
        val rh = y1 - y0

        val ref = Bitmap.createBitmap(rw, rh, Bitmap.Config.ARGB_8888)
        try {
            Canvas(ref).drawBitmap(
                source,
                Rect(box.left, box.top, box.right, box.bottom),
                Rect(0, 0, rw, rh),
                Paint(Paint.FILTER_BITMAP_FLAG)
            )
            val a = IntArray(rw * rh)
            val b = IntArray(rw * rh)
            result.getPixels(a, 0, rw, x0, y0, rw, rh)
            ref.getPixels(b, 0, rw, 0, 0, rw, rh)
            for (i in a.indices) {
                val p = a[i]
                val q = b[i]
                a[i] = (0xFF shl 24) or
                    (mix((p shr 16) and 0xFF, (q shr 16) and 0xFF, weight) shl 16) or
                    (mix((p shr 8) and 0xFF, (q shr 8) and 0xFF, weight) shl 8) or
                    mix(p and 0xFF, q and 0xFF, weight)
            }
            result.setPixels(a, 0, rw, x0, y0, rw, rh)
        } finally {
            ref.recycle()
        }
    }

    /** Mean gradient magnitude of the luminance plane: a detail proxy. */
    private fun detailEnergy(px: IntArray, w: Int, h: Int): Float {
        var sum = 0.0
        var n = 0
        for (y in 1 until h - 1) {
            val base = y * w
            for (x in 1 until w - 1) {
                val i = base + x
                val dx = luma(px[i + 1]) - luma(px[i - 1])
                val dy = luma(px[i + w]) - luma(px[i - w])
                sum += sqrt((dx * dx + dy * dy).toDouble())
                n++
            }
        }
        return if (n == 0) 0f else (sum / n).toFloat()
    }

    /** Pearson correlation of the two luminance planes. */
    private fun correlation(a: IntArray, b: IntArray, n: Int): Float {
        var sa = 0.0; var sb = 0.0
        for (i in 0 until n) { sa += luma(a[i]); sb += luma(b[i]) }
        val ma = sa / n
        val mb = sb / n
        var cov = 0.0; var va = 0.0; var vb = 0.0
        for (i in 0 until n) {
            val da = luma(a[i]) - ma
            val db = luma(b[i]) - mb
            cov += da * db; va += da * da; vb += db * db
        }
        if (va <= 1e-6 || vb <= 1e-6) return 1f
        return (cov / (sqrt(va) * sqrt(vb))).toFloat()
    }

    private fun luma(c: Int): Float =
        0.2126f * ((c shr 16) and 0xFF) +
            0.7152f * ((c shr 8) and 0xFF) +
            0.0722f * (c and 0xFF)

    private fun mix(a: Int, b: Int, w: Float): Int {
        val v = (a + (b - a) * w).roundToInt()
        return if (v < 0) 0 else if (v > 255) 255 else v
    }

    override fun close() {
        runCatching { session?.close() }
    }
}
