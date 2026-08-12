package com.photoenhancer.editor

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.content.Context
import android.graphics.Bitmap
import java.io.Closeable
import java.nio.FloatBuffer
import kotlin.math.abs
import kotlin.math.max
import kotlin.math.min
import kotlin.math.roundToInt

/**
 * Stage 2: repair the source *before* it is enlarged.
 *
 * Order matters and is not arbitrary. Noise and JPEG blocking are both
 * high-frequency corruption, and x4 super-resolution treats any
 * high-frequency structure as detail worth amplifying — so leaving them in
 * means the SR model faithfully enlarges the damage. Deblurring is applied
 * last, because sharpening noise first would only amplify it.
 *
 *  - Denoise + deblock: SCUNet (fp16, 40 MB). Measured +4.47 dB PSNR on a
 *    real noise + JPEG q30 degradation, so one model covers both defects.
 *  - Deblur: classical constrained sharpening, gated by an edge mask and
 *    steered by the measured blur direction. A learned deblurrer was
 *    evaluated and rejected: NAFNet only accepts 512x512, costs 7.6 s per
 *    tile, and its fp16 conversion produces NaN.
 *
 * SCUNet is fully convolutional, but its internal 8x down/up path means the
 * input must be a multiple of 64, so tiles are padded by reflection and the
 * padding is discarded afterwards.
 */
class CleanupStage private constructor(
    private val env: OrtEnvironment,
    private val session: OrtSession
) : Closeable {

    companion object {
        private const val ASSET_SOTA = "nafnet.onnx"
        private const val ASSET_FALLBACK = "scunet_fp16.onnx"

        /** SCUNet's stride requirement; tiles are padded up to this grid. */
        private const val ALIGN = 64

        /** Tile side for cleanup. Large enough to keep context, small enough
         *  that a padded tile stays a few MB of float. */
        const val TILE = 256

        /** Overlap trimmed from each side, hiding tile boundaries entirely. */
        const val MARGIN = 16

        fun create(ctx: Context, env: OrtEnvironment, threads: Int): CleanupStage? = try {
            val asset = if (ModelStore.isStaged(ctx, ASSET_SOTA) || ModelStore.fileOf(ctx, ASSET_SOTA).exists()) {
                ASSET_SOTA
            } else {
                ASSET_FALLBACK
            }
            val path = ModelStore.ensure(ctx, asset)
            val label = if (asset == ASSET_SOTA) "تنظيف SOTA (NAFNet)" else "تنظيف (SCUNet)"
            val s = Accelerator.create(
                env = env,
                modelPath = path,
                label = label,
                threads = threads,
                precision = Accelerator.Precision.RELAXED,
                verify = { session -> probe(env, session) }
            )
            if (s == null) null else CleanupStage(env, s)
        } catch (t: Throwable) {
            null
        }

        /**
         * Runs one small aligned tile and refuses the session unless the result
         * is finite and in range. A driver can accept the graph and still
         * return NaN, so acceptance has to be earned on real data.
         */
        private fun probe(env: OrtEnvironment, s: OrtSession): Boolean {
            return try {
                val n = 3 * ALIGN * ALIGN
                val buf = java.nio.FloatBuffer.allocate(n)
                val arr = buf.array()
                for (i in arr.indices) arr[i] = ((i * 31) % 255) / 255f
                val shape = longArrayOf(1, 3, ALIGN.toLong(), ALIGN.toLong())
                val inName = s.inputNames.iterator().next()
                buf.rewind()
                OnnxTensor.createTensor(env, buf, shape).use { t ->
                    s.run(mapOf(inName to t)).use { res ->
                        val ob = (res[0] as OnnxTensor).floatBuffer
                        var i = 0
                        while (i < ob.capacity()) {
                            val v = ob.get(i)
                            if (v.isNaN() || v.isInfinite() || v < -0.5f || v > 1.5f) return false
                            i += 617   // prime stride: no channel is skipped
                        }
                    }
                }
                true
            } catch (t: Throwable) {
                false
            }
        }
    }

    /** Number of neural tiles a cleanup pass over w x h will run. */
    fun tileCount(w: Int, h: Int): Int {
        val step = TILE - 2 * MARGIN
        val cols = (w + step - 1) / step
        val rows = (h + step - 1) / step
        return max(1, cols) * max(1, rows)
    }

    /**
     * Runs the needed repairs and returns a new bitmap. When nothing is
     * needed the input is returned unchanged (caller must not assume a copy).
     *
     * @param onTile invoked after each neural tile, for progress reporting.
     */
    fun apply(
        src: Bitmap,
        report: QualityAnalyzer.Report,
        governor: ThermalGovernor?,
        cancelled: () -> Boolean,
        onTile: (Long) -> Unit
    ): Bitmap {
        var current = src

        if (report.needsDenoise || report.needsDeblock) {
            val cleaned = denoise(current, governor, cancelled, onTile)
            if (cleaned != null) {
                if (current !== src) current.recycle()
                current = cleaned
            }
        }
        if (cancelled()) return current

        if (report.needsDeblur) {
            val sharp = deblur(current, report)
            if (sharp != null) {
                if (current !== src) current.recycle()
                current = sharp
            }
        }
        return current
    }

    // --------------------------------------------------------------- denoise

    private fun denoise(
        src: Bitmap,
        governor: ThermalGovernor?,
        cancelled: () -> Boolean,
        onTile: (Long) -> Unit
    ): Bitmap? {
        return try {
            val w = src.width
            val h = src.height
            val out = Bitmap.createBitmap(w, h, Bitmap.Config.ARGB_8888)
            val step = TILE - 2 * MARGIN
            val inName = session.inputNames.iterator().next()

            var ty = 0
            while (ty < h) {
                var tx = 0
                while (tx < w) {
                    if (cancelled()) { out.recycle(); return null }
                    val t0 = System.currentTimeMillis()

                    // Region we will read (with margin) and the part we keep.
                    val rx0 = max(0, tx - MARGIN)
                    val ry0 = max(0, ty - MARGIN)
                    val rx1 = min(w, tx + step + MARGIN)
                    val ry1 = min(h, ty + step + MARGIN)
                    val rw = rx1 - rx0
                    val rh = ry1 - ry0

                    // Pad up to the model's 64-pixel grid by edge replication.
                    val pw = ((rw + ALIGN - 1) / ALIGN) * ALIGN
                    val ph = ((rh + ALIGN - 1) / ALIGN) * ALIGN

                    val buf = FloatBuffer.allocate(3 * pw * ph)
                    val arr = buf.array()
                    val plane = pw * ph
                    val row = IntArray(rw)
                    for (y in 0 until ph) {
                        val sy = ry0 + min(y, rh - 1)
                        src.getPixels(row, 0, rw, rx0, sy, rw, 1)
                        val base = y * pw
                        for (x in 0 until pw) {
                            val c = row[min(x, rw - 1)]
                            arr[base + x] = ((c shr 16) and 0xFF) / 255f
                            arr[plane + base + x] = ((c shr 8) and 0xFF) / 255f
                            arr[2 * plane + base + x] = (c and 0xFF) / 255f
                        }
                    }
                    buf.rewind()

                    val shape = longArrayOf(1, 3, ph.toLong(), pw.toLong())
                    OnnxTensor.createTensor(env, buf, shape).use { tensor ->
                        session.run(mapOf(inName to tensor)).use { res ->
                            val ob = (res[0] as OnnxTensor).floatBuffer
                            // Write back only the interior, without the margin.
                            val kx0 = tx
                            val ky0 = ty
                            val kx1 = min(w, tx + step)
                            val ky1 = min(h, ty + step)
                            val kw = kx1 - kx0
                            val kh = ky1 - ky0
                            if (kw > 0 && kh > 0) {
                                val px = IntArray(kw * kh)
                                for (y in 0 until kh) {
                                    val sy = (ky0 - ry0) + y
                                    for (x in 0 until kw) {
                                        val sx = (kx0 - rx0) + x
                                        val i = sy * pw + sx
                                        px[y * kw + x] = (0xFF shl 24) or
                                            (unit(ob.get(i)) shl 16) or
                                            (unit(ob.get(plane + i)) shl 8) or
                                            unit(ob.get(2 * plane + i))
                                    }
                                }
                                out.setPixels(px, 0, kw, kx0, ky0, kw, kh)
                            }
                        }
                    }

                    val ms = System.currentTimeMillis() - t0
                    onTile(ms)
                    governor?.poll()
                    governor?.coolDown(ms) { cancelled() }
                    tx += step
                }
                ty += step
            }
            out
        } catch (t: Throwable) {
            null
        }
    }

    // ---------------------------------------------------------------- deblur

    /**
     * Edge-gated directional sharpening.
     *
     * This is intentionally conservative: it recovers acutance that blur took
     * away without inventing structure. The correction is only applied where a
     * real edge exists (so flat sky and skin are left alone), and for motion
     * blur the difference is taken *across* the measured smear direction,
     * which is where the information was actually lost.
     */
    private fun deblur(src: Bitmap, report: QualityAnalyzer.Report): Bitmap? {
        return try {
            val w = src.width
            val h = src.height
            if (w < 5 || h < 5) return null

            val px = IntArray(w * h)
            src.getPixels(px, 0, w, 0, 0, w, h)
            val out = IntArray(w * h)

            // Blur amount drives the gain, capped so heavy blur cannot ring.
            val gain = min(1.15f, 0.55f + report.blurScore * 0.9f)

            // Sample offsets: perpendicular to the smear for motion blur, and a
            // symmetric cross for defocus.
            val (ax, ay) = when (report.blurAngleDeg) {
                0 -> Pair(0, 2)      // horizontal smear -> correct vertically
                90 -> Pair(2, 0)
                45 -> Pair(-2, 2)
                135 -> Pair(2, 2)
                else -> Pair(0, 0)
            }

            val luma = FloatArray(w * h)
            for (i in px.indices) {
                val c = px[i]
                luma[i] = 0.2126f * ((c shr 16) and 0xFF) +
                    0.7152f * ((c shr 8) and 0xFF) +
                    0.0722f * (c and 0xFF)
            }

            for (y in 0 until h) {
                val base = y * w
                for (x in 0 until w) {
                    val i = base + x
                    // Local edge strength decides whether to touch this pixel.
                    val xm = if (x > 0) i - 1 else i
                    val xp = if (x < w - 1) i + 1 else i
                    val ym = if (y > 0) i - w else i
                    val yp = if (y < h - 1) i + w else i
                    val edge = abs(luma[xp] - luma[xm]) + abs(luma[yp] - luma[ym])
                    // 6 gray levels of gradient = a definite edge; below ~2 it is
                    // flat area or noise and must stay untouched.
                    val mask = ((edge - 2f) / 6f).coerceIn(0f, 1f)
                    if (mask <= 0.01f) { out[i] = px[i]; continue }

                    val blurAt = if (ax == 0 && ay == 0) {
                        // Defocus: symmetric 5-tap cross average.
                        avgOf(px, w, h, x, y, intArrayOf(-2, 0, 2, 0, 0), intArrayOf(0, -2, 0, 2, 0))
                    } else {
                        avgOf(px, w, h, x, y, intArrayOf(-ax, ax, 0), intArrayOf(-ay, ay, 0))
                    }
                    val s = px[i]
                    val g = gain * mask
                    out[i] = (0xFF shl 24) or
                        (push((s shr 16) and 0xFF, (blurAt shr 16) and 0xFF, g) shl 16) or
                        (push((s shr 8) and 0xFF, (blurAt shr 8) and 0xFF, g) shl 8) or
                        push(s and 0xFF, blurAt and 0xFF, g)
                }
            }
            val bmp = Bitmap.createBitmap(w, h, Bitmap.Config.ARGB_8888)
            bmp.setPixels(out, 0, w, 0, 0, w, h)
            bmp
        } catch (t: Throwable) {
            null
        }
    }

    private fun avgOf(px: IntArray, w: Int, h: Int, x: Int, y: Int, dxs: IntArray, dys: IntArray): Int {
        var r = 0; var g = 0; var b = 0
        for (k in dxs.indices) {
            val sx = (x + dxs[k]).coerceIn(0, w - 1)
            val sy = (y + dys[k]).coerceIn(0, h - 1)
            val c = px[sy * w + sx]
            r += (c shr 16) and 0xFF
            g += (c shr 8) and 0xFF
            b += c and 0xFF
        }
        val n = dxs.size
        return ((r / n) shl 16) or ((g / n) shl 8) or (b / n)
    }

    private fun push(s: Int, b: Int, gain: Float): Int {
        val v = s + ((s - b) * gain).roundToInt()
        return if (v < 0) 0 else if (v > 255) 255 else v
    }

    private fun unit(v: Float): Int {
        val x = (v * 255f).roundToInt()
        return if (x < 0) 0 else if (x > 255) 255 else x
    }

    override fun close() {
        runCatching { session.close() }
    }
}
