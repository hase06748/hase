package com.photoenhancer.editor

import android.graphics.Bitmap
import kotlin.math.abs
import kotlin.math.max
import kotlin.math.min
import kotlin.math.sqrt

/**
 * Stage 1 of the pipeline: a fast, purely classical read of *how* the picture
 * is damaged, so the later stages only run when they are actually needed.
 *
 * Everything here is measured on a luminance plane sub-sampled to at most
 * ~256k pixels, so the whole analysis costs a few milliseconds even for a
 * 50 MP source. Four questions get answered:
 *
 *   - ISO noise?      -> median absolute deviation of a high-pass residual,
 *                        taken over the *flattest* tiles only, so texture is
 *                        not mistaken for grain.
 *   - JPEG artifacts? -> block-boundary energy at the 8-pixel DCT grid versus
 *                        the energy at non-boundary offsets. A clean image has
 *                        no reason to prefer multiples of 8.
 *   - Blur, and which kind? -> Tenengrad focus score for the amount, plus the
 *                        anisotropy of directional gradients for the type:
 *                        motion blur smears one direction and leaves the
 *                        perpendicular one sharp, defocus dulls both equally.
 *   - Faces?          -> delegated to [FaceDetector] (YuNet).
 */
object QualityAnalyzer {

    /** Roughly how many pixels the analysis plane is allowed to hold. */
    private const val MAX_ANALYSIS_PIXELS = 262_144

    enum class BlurKind { NONE, MOTION, DEFOCUS }

    /**
     * @param noiseSigma  estimated Gaussian sigma in 0..255 units.
     * @param jpegScore   0 = no blocking, 1 = severe blocking.
     * @param blurScore   0 = tack sharp, 1 = heavily blurred.
     * @param blurKind    which kind of blur dominates, if any.
     * @param blurAngleDeg dominant smear direction for motion blur, else -1.
     * @param faces       face boxes in *source* pixel coordinates.
     */
    data class Report(
        val noiseSigma: Float,
        val jpegScore: Float,
        val blurScore: Float,
        val blurKind: BlurKind,
        val blurAngleDeg: Int,
        val faces: List<FaceDetector.Face>,
        val analysisMs: Long
    ) {
        val needsDenoise: Boolean get() = noiseSigma > 3.0f
        val needsDeblock: Boolean get() = jpegScore > 0.18f
        val needsDeblur: Boolean get() = blurKind != BlurKind.NONE && blurScore > 0.35f
        val hasFaces: Boolean get() = faces.isNotEmpty()

        /** Short Arabic summary for the UI. */
        fun summary(): String {
            val parts = ArrayList<String>(4)
            if (needsDenoise) parts.add("تشويش ${noiseSigma.toInt()}")
            if (needsDeblock) parts.add("آثار JPEG")
            if (needsDeblur) {
                parts.add(if (blurKind == BlurKind.MOTION) "ضباب حركة" else "خارج التركيز")
            }
            if (hasFaces) parts.add("${faces.size} وجه")
            return if (parts.isEmpty()) "الصورة سليمة" else parts.joinToString(" · ")
        }
    }

    /** Luminance plane plus its dimensions. */
    private class Plane(val v: FloatArray, val w: Int, val h: Int)

    /**
     * @param src      the image the later stages will actually work on; faces
     *                 are reported in *its* coordinate space.
     * @param native   the untouched, full-resolution decode, when one exists.
     *                 Noise and blocking are read from here — see below.
     */
    fun analyze(src: Bitmap, detector: FaceDetector?, native: Bitmap? = null): Report {
        val t0 = System.currentTimeMillis()
        val plane = luma(src)

        // Noise and JPEG blocking are the two measurements that cannot survive
        // a resample. A smooth downscale low-passes exactly the grain we are
        // looking for, and integer decimation by a step that is not a divisor
        // of 8 (step 7 for a 4000x3000 source) shreds the DCT grid the
        // blocking metric depends on. Both are therefore read from small,
        // 8-aligned windows taken at native resolution.
        val windows = nativeWindows(native ?: src)
        val noise: Float
        val jpeg: Float
        if (windows.isEmpty()) {
            noise = estimateNoise(plane)
            jpeg = estimateBlocking(plane)
        } else {
            noise = median(FloatArray(windows.size) { estimateNoise(windows[it]) })
            jpeg = median(FloatArray(windows.size) { estimateBlocking(windows[it]) })
        }
        val (blurScore, kind, angle) = estimateBlur(plane, noise)
        val faces = detector?.let { runCatching { it.detect(src) }.getOrDefault(emptyList()) }
            ?: emptyList()

        return Report(
            noiseSigma = noise,
            jpegScore = jpeg,
            blurScore = blurScore,
            blurKind = kind,
            blurAngleDeg = angle,
            faces = faces,
            analysisMs = System.currentTimeMillis() - t0
        )
    }

    // ------------------------------------------------------------- luminance

    /**
     * Rec.709 luminance, decimated by an integer step. Integer decimation is
     * deliberate: a smooth resample would low-pass the very noise and blocking
     * we are trying to measure.
     */
    private fun luma(src: Bitmap): Plane {
        val sw = src.width
        val sh = src.height
        var step = 1
        while ((sw / step).toLong() * (sh / step) > MAX_ANALYSIS_PIXELS) step++

        val w = max(8, sw / step)
        val h = max(8, sh / step)
        val out = FloatArray(w * h)
        val row = IntArray(sw)
        for (y in 0 until h) {
            src.getPixels(row, 0, sw, 0, min(y * step, sh - 1), sw, 1)
            val base = y * w
            for (x in 0 until w) {
                val c = row[min(x * step, sw - 1)]
                val r = (c shr 16) and 0xFF
                val g = (c shr 8) and 0xFF
                val b = c and 0xFF
                out[base + x] = 0.2126f * r + 0.7152f * g + 0.0722f * b
            }
        }
        return Plane(out, w, h)
    }

    /** Side of each native-resolution probe window, a multiple of 8. */
    private const val WIN = 256

    /** Probe windows per axis; 3x3 = 9 samples, ~590k pixels read in total. */
    private const val GRID = 3

    /**
     * Up to GRID x GRID luminance windows lifted straight out of the source at
     * 1:1 scale, each one starting on a multiple of 8 so JPEG's block grid
     * stays where the encoder put it. Sampling several spread-out windows and
     * taking the median afterwards keeps one unlucky patch — a flat sky, a
     * busy fabric — from deciding the whole verdict.
     */
    private fun nativeWindows(src: Bitmap): List<Plane> {
        val sw = src.width
        val sh = src.height
        if (sw < 64 || sh < 64) return emptyList()

        val w = min(WIN, sw and 0xFFFFFFF8.toInt())
        val h = min(WIN, sh and 0xFFFFFFF8.toInt())
        if (w < 32 || h < 32) return emptyList()

        val xs = originsOn8(sw, w)
        val ys = originsOn8(sh, h)
        val out = ArrayList<Plane>(xs.size * ys.size)
        val row = IntArray(w)

        for (oy in ys) {
            for (ox in xs) {
                val v = FloatArray(w * h)
                for (y in 0 until h) {
                    src.getPixels(row, 0, w, ox, oy + y, w, 1)
                    val base = y * w
                    for (x in 0 until w) {
                        val c = row[x]
                        val r = (c shr 16) and 0xFF
                        val g = (c shr 8) and 0xFF
                        val b = c and 0xFF
                        v[base + x] = 0.2126f * r + 0.7152f * g + 0.0722f * b
                    }
                }
                out.add(Plane(v, w, h))
            }
        }
        return out
    }

    /** Evenly spread window origins, each snapped down to a multiple of 8. */
    private fun originsOn8(size: Int, win: Int): IntArray {
        if (size <= win) return intArrayOf(0)
        val span = size - win
        val n = min(GRID, span / 8 + 1)
        return IntArray(n) { i ->
            val raw = if (n == 1) span / 2 else span * i / (n - 1)
            (raw / 8) * 8
        }
    }

    // ----------------------------------------------------------------- noise

    /**
     * Sigma from the flattest 25% of 16x16 tiles. Within each tile we take the
     * MAD of a Laplacian residual; MAD is used instead of a plain standard
     * deviation because it ignores the few strong edges that survive even in a
     * flat tile. 1.4826 converts MAD to sigma for Gaussian data, and 0.25 is
     * the gain of the 4-neighbour Laplacian kernel.
     */
    private fun estimateNoise(p: Plane): Float {
        val tile = 16
        if (p.w < tile * 2 || p.h < tile * 2) return 0f
        val cols = p.w / tile
        val rows = p.h / tile
        val sigmas = ArrayList<Float>(cols * rows)
        val buf = FloatArray(tile * tile)

        for (ty in 0 until rows) {
            for (tx in 0 until cols) {
                var n = 0
                for (y in 1 until tile - 1) {
                    val gy = ty * tile + y
                    for (x in 1 until tile - 1) {
                        val gx = tx * tile + x
                        val i = gy * p.w + gx
                        val lap = 4f * p.v[i] - p.v[i - 1] - p.v[i + 1] -
                            p.v[i - p.w] - p.v[i + p.w]
                        buf[n++] = lap
                    }
                }
                if (n < 16) continue
                val slice = buf.copyOf(n)
                val med = median(slice)
                for (i in 0 until n) slice[i] = abs(slice[i] - med)
                sigmas.add(median(slice) * 1.4826f * 0.25f)
            }
        }
        if (sigmas.isEmpty()) return 0f
        sigmas.sort()
        // Flat tiles cluster at the bottom; take their median.
        val keep = max(1, sigmas.size / 4)
        var sum = 0f
        for (i in 0 until keep) sum += sigmas[i]
        return sum / keep
    }

    // ------------------------------------------------------------ jpeg blocks

    /**
     * JPEG quantises 8x8 blocks independently, which leaves a step at every
     * multiple of 8. We compare the mean absolute first difference *at* those
     * columns/rows with the mean at all other offsets. A ratio above 1 means
     * the 8-grid is special, i.e. blocking.
     */
    private fun estimateBlocking(p: Plane): Float {
        if (p.w < 24 || p.h < 24) return 0f

        var onEdge = 0.0
        var onEdgeN = 0
        var offEdge = 0.0
        var offEdgeN = 0

        for (y in 1 until p.h - 1) {
            val base = y * p.w
            for (x in 8 until p.w - 1 step 1) {
                val d = abs(p.v[base + x] - p.v[base + x - 1]).toDouble()
                if (x % 8 == 0) { onEdge += d; onEdgeN++ } else { offEdge += d; offEdgeN++ }
            }
        }
        for (x in 1 until p.w - 1) {
            for (y in 8 until p.h - 1 step 1) {
                val d = abs(p.v[y * p.w + x] - p.v[(y - 1) * p.w + x]).toDouble()
                if (y % 8 == 0) { onEdge += d; onEdgeN++ } else { offEdge += d; offEdgeN++ }
            }
        }
        if (onEdgeN == 0 || offEdgeN == 0) return 0f
        val on = onEdge / onEdgeN
        val off = offEdge / offEdgeN
        if (off < 1e-4) return 0f
        // ratio 1.0 -> no blocking, 1.5+ -> obvious. Map to 0..1.
        val ratio = (on / off).toFloat()
        return min(1f, max(0f, (ratio - 1.0f) / 0.6f))
    }

    // ------------------------------------------------------------------ blur

    /**
     * Returns (score, kind, angle).
     *
     * The amount comes from a Tenengrad-style mean gradient magnitude,
     * normalised against the noise floor so a noisy-but-sharp image is not
     * called blurry. The *kind* comes from comparing gradient energy along
     * four directions: motion blur kills one axis and spares its
     * perpendicular, so the min/max ratio drops well below 1; defocus
     * attenuates every direction alike and keeps the ratio near 1.
     */
    private fun estimateBlur(p: Plane, noiseSigma: Float): Triple<Float, BlurKind, Int> {
        if (p.w < 8 || p.h < 8) return Triple(0f, BlurKind.NONE, -1)

        var gx = 0.0
        var gy = 0.0
        var gd1 = 0.0
        var gd2 = 0.0
        var mag = 0.0
        var n = 0

        for (y in 1 until p.h - 1) {
            val base = y * p.w
            val up = base - p.w
            val dn = base + p.w
            for (x in 1 until p.w - 1) {
                val dx = p.v[base + x + 1] - p.v[base + x - 1]
                val dy = p.v[dn + x] - p.v[up + x]
                val d1 = p.v[dn + x + 1] - p.v[up + x - 1]   // 45 degrees
                val d2 = p.v[dn + x - 1] - p.v[up + x + 1]   // 135 degrees
                gx += abs(dx); gy += abs(dy)
                gd1 += abs(d1); gd2 += abs(d2)
                mag += sqrt((dx * dx + dy * dy).toDouble())
                n++
            }
        }
        if (n == 0) return Triple(0f, BlurKind.NONE, -1)

        val meanMag = (mag / n).toFloat()
        // Subtract the gradient a pure noise field would produce (~1.6 sigma
        // for a 2-tap central difference on both axes) before judging focus.
        val focus = max(0f, meanMag - 1.6f * noiseSigma)
        // 12 gray levels of mean gradient is a comfortably sharp photo.
        val score = 1f - min(1f, focus / 12f)

        val dirs = doubleArrayOf(gx / n, gy / n, gd1 / (n * 1.414), gd2 / (n * 1.414))
        var lo = 0
        var hi = 0
        for (i in dirs.indices) {
            if (dirs[i] < dirs[lo]) lo = i
            if (dirs[i] > dirs[hi]) hi = i
        }
        val aniso = if (dirs[hi] > 1e-6) (dirs[lo] / dirs[hi]).toFloat() else 1f

        val kind = when {
            score < 0.35f -> BlurKind.NONE
            aniso < 0.55f -> BlurKind.MOTION
            else -> BlurKind.DEFOCUS
        }
        // The smear runs *perpendicular* to the surviving gradient direction,
        // i.e. along the axis whose gradient collapsed.
        val angle = if (kind != BlurKind.MOTION) -1 else when (lo) {
            0 -> 90    // horizontal gradient lost -> vertical detail gone
            1 -> 0
            2 -> 135
            else -> 45
        }
        return Triple(score, kind, angle)
    }

    // --------------------------------------------------------------- helpers

    private fun median(a: FloatArray): Float {
        a.sort()
        val n = a.size
        return if (n == 0) 0f
        else if (n % 2 == 1) a[n / 2]
        else 0.5f * (a[n / 2 - 1] + a[n / 2])
    }
}
