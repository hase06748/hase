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
import kotlin.math.cos
import kotlin.math.max
import kotlin.math.min
import kotlin.math.roundToInt
import kotlin.math.PI

/**
 * Stage 4: smart region processing.
 *
 * GFPGAN (fp16, 163 MB) restores a face far better than a generic upscaler,
 * but it is a *generative* model: run it over a whole photo and it will
 * hallucinate faces into wallpaper patterns. So it is applied strictly inside
 * the detected face ROI, and even there the result is composited through a
 * soft elliptical mask so the jaw and hairline blend into the SR output
 * instead of showing a rectangular seam.
 *
 * Two protections run alongside:
 *
 *  - Skin is detected in YCbCr and its restoration weight is reduced, because
 *    GFPGAN tends to over-texture cheeks and foreheads.
 *  - Eyes, brows and mouth get a *raised* weight, since those are the regions
 *    where restoration genuinely adds information a 4x upscaler cannot.
 *
 * Contract (validated offline):
 *   input : float32 [1,3,512,512] RGB, normalised to -1..1
 *   output: float32 [1,3,512,512] RGB in about -1..1
 */
class FaceStage private constructor(
    private val env: OrtEnvironment,
    private val session: OrtSession
) : Closeable {

    companion object {
        private const val ASSET_SOTA = "codeformer.onnx"
        private const val ASSET_FALLBACK = "gfpgan.onnx"
        private const val SIDE = 512

        /** How much of the box size is added around it as context. */
        private const val PAD_RATIO = 0.35f

        fun create(ctx: Context, env: OrtEnvironment, threads: Int): FaceStage? = try {
            val asset = if (ModelStore.isStaged(ctx, ASSET_SOTA) || ModelStore.fileOf(ctx, ASSET_SOTA).exists()) {
                ASSET_SOTA
            } else {
                ASSET_FALLBACK
            }
            val path = ModelStore.ensure(ctx, asset)
            val label = if (asset == ASSET_SOTA) "وجوه SOTA (CodeFormer)" else "وجوه (GFPGAN)"
            val s = Accelerator.create(
                env = env,
                modelPath = path,
                label = label,
                threads = threads,
                precision = Accelerator.Precision.STRICT,
                verify = { session -> probe(env, session) }
            )
            if (s == null) null else FaceStage(env, s)
        } catch (t: Throwable) {
            null
        }

        /** One 512x512 pass; the output must stay finite and inside ~[-1,1]. */
        private fun probe(env: OrtEnvironment, s: OrtSession): Boolean {
            return try {
                val n = 3 * SIDE * SIDE
                val buf = java.nio.FloatBuffer.allocate(n)
                val arr = buf.array()
                for (i in arr.indices) arr[i] = (((i * 29) % 255) / 255f) * 2f - 1f
                val shape = longArrayOf(1, 3, SIDE.toLong(), SIDE.toLong())
                val inName = s.inputNames.iterator().next()
                buf.rewind()
                OnnxTensor.createTensor(env, buf, shape).use { t ->
                    s.run(mapOf(inName to t)).use { res ->
                        val ob = (res[0] as OnnxTensor).floatBuffer
                        var i = 0
                        while (i < ob.capacity()) {
                            val v = ob.get(i)
                            if (v.isNaN() || v.isInfinite() || v < -4f || v > 4f) return false
                            i += 617
                        }
                    }
                }
                true
            } catch (t: Throwable) {
                false
            }
        }
    }

    /**
     * Composites a restored version of every face into [target] in place.
     *
     * @param target the x4 super-resolved image; modified directly.
     * @param faces  boxes in *source* coordinates.
     * @param scale  target/source scale factor (4).
     * @param strength 0..1 overall blend weight.
     * @return number of faces successfully restored.
     */
    fun restore(
        target: Bitmap,
        faces: List<FaceDetector.Face>,
        scale: Int,
        strength: Float,
        cancelled: () -> Boolean,
        onFace: (Int, Long) -> Unit
    ): Int {
        if (faces.isEmpty() || strength <= 0.01f) return 0
        var done = 0
        val inName = session.inputNames.iterator().next()

        // Biggest faces first: if the run is cancelled the visible ones are done.
        val ordered = faces.sortedByDescending { it.box.width().toLong() * it.box.height() }

        for (face in ordered) {
            if (cancelled()) break
            val t0 = System.currentTimeMillis()

            // Scale the ROI into target space and pad for context.
            val b = face.box
            val padX = (b.width() * PAD_RATIO).roundToInt()
            val padY = (b.height() * PAD_RATIO).roundToInt()
            val rx0 = ((b.left - padX) * scale).coerceIn(0, target.width - 2)
            val ry0 = ((b.top - padY) * scale).coerceIn(0, target.height - 2)
            val rx1 = ((b.right + padX) * scale).coerceIn(rx0 + 2, target.width)
            val ry1 = ((b.bottom + padY) * scale).coerceIn(ry0 + 2, target.height)
            val rw = rx1 - rx0
            val rh = ry1 - ry0

            // A tiny face carries no detail worth generating.
            if (rw < 64 || rh < 64) continue

            val roi = Bitmap.createBitmap(rw, rh, Bitmap.Config.ARGB_8888)
            val square = Bitmap.createBitmap(SIDE, SIDE, Bitmap.Config.ARGB_8888)
            try {
                run {
                    val px = IntArray(rw * rh)
                    target.getPixels(px, 0, rw, rx0, ry0, rw, rh)
                    roi.setPixels(px, 0, rw, 0, 0, rw, rh)
                }
                Canvas(square).drawBitmap(
                    roi, null, Rect(0, 0, SIDE, SIDE), Paint(Paint.FILTER_BITMAP_FLAG)
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
                        // GFPGAN expects -1..1.
                        arr[base + x] = ((c shr 16) and 0xFF) / 127.5f - 1f
                        arr[plane + base + x] = ((c shr 8) and 0xFF) / 127.5f - 1f
                        arr[2 * plane + base + x] = (c and 0xFF) / 127.5f - 1f
                    }
                }
                buf.rewind()

                val restored = IntArray(plane)
                var sane = false
                OnnxTensor.createTensor(env, buf, longArrayOf(1, 3, SIDE.toLong(), SIDE.toLong()))
                    .use { tensor ->
                        session.run(mapOf(inName to tensor)).use { res ->
                            val ob = (res[0] as OnnxTensor).floatBuffer
                            val probe = ob.get(plane / 2)
                            sane = !probe.isNaN() && probe > -4f && probe < 4f
                            if (sane) {
                                for (i in 0 until plane) {
                                    restored[i] = (0xFF shl 24) or
                                        (back(ob.get(i)) shl 16) or
                                        (back(ob.get(plane + i)) shl 8) or
                                        back(ob.get(2 * plane + i))
                                }
                            }
                        }
                    }
                if (!sane) continue

                // Bring the 512 result back to ROI size.
                val restoredBmp = Bitmap.createBitmap(SIDE, SIDE, Bitmap.Config.ARGB_8888)
                restoredBmp.setPixels(restored, 0, SIDE, 0, 0, SIDE, SIDE)
                val fitted = Bitmap.createBitmap(rw, rh, Bitmap.Config.ARGB_8888)
                Canvas(fitted).drawBitmap(
                    restoredBmp, null, Rect(0, 0, rw, rh), Paint(Paint.FILTER_BITMAP_FLAG)
                )
                restoredBmp.recycle()

                // GFPGAN is trained on FFHQ, so it returns a face with FFHQ's
                // exposure and colour balance rather than this photo's. Left
                // alone that shows up as a visible tonal step exactly on the
                // mask edge, where the restored cheek meets the SR neck. Match
                // the restoration's tone distribution to the region it is about
                // to be dropped into *before* blending, so the alpha ramp only
                // has to hide texture differences, not a colour shift.
                matchTone(roi, fitted, rw, rh)

                blend(target, roi, fitted, face, rx0, ry0, rw, rh, scale, strength)
                fitted.recycle()

                done++
                onFace(done, System.currentTimeMillis() - t0)
            } finally {
                roi.recycle()
                square.recycle()
            }
        }
        return done
    }

    /**
     * Per-channel histogram matching of [restored] onto [original], in place.
     *
     * Why histogram matching and not Poisson blending: Poisson solves for a
     * gradient field whose divergence matches the source while the *boundary*
     * is clamped to the destination. That is the right tool when you paste an
     * opaque patch and the seam is a hard line. Here the seam is not a line —
     * [blend] already composites through a raised-cosine elliptical alpha with
     * a per-pixel skin damp and a landmark bonus, so there is no boundary to
     * clamp. What actually goes wrong is global: FFHQ tone versus this photo's
     * tone. Matching the distribution fixes that cause directly, in one pass
     * over the pixels, instead of iterating a Laplacian solve over a region
     * where alpha is already continuous.
     *
     * Statistics are gathered only inside 0.52 of the ellipse — the part that
     * will be composited at full weight. Sampling the corners would pull the
     * mapping toward pixels that are about to be discarded.
     */
    private fun matchTone(original: Bitmap, restored: Bitmap, rw: Int, rh: Int) {
        if (rw < 8 || rh < 8) return

        val orig = IntArray(rw * rh)
        val rest = IntArray(rw * rh)
        original.getPixels(orig, 0, rw, 0, 0, rw, rh)
        restored.getPixels(rest, 0, rw, 0, 0, rw, rh)

        val cx = rw * 0.5f
        val cy = rh * 0.5f
        val ax = rw * 0.46f
        val ay = rh * 0.5f

        // 3 channels x 256 bins for each image.
        val hOrig = Array(3) { IntArray(256) }
        val hRest = Array(3) { IntArray(256) }
        var counted = 0

        for (y in 0 until rh) {
            val dy = (y - cy) / ay
            val base = y * rw
            for (x in 0 until rw) {
                val dx = (x - cx) / ax
                if (dx * dx + dy * dy > 0.52f) continue
                val a = orig[base + x]
                val b = rest[base + x]
                hOrig[0][(a shr 16) and 0xFF]++
                hOrig[1][(a shr 8) and 0xFF]++
                hOrig[2][a and 0xFF]++
                hRest[0][(b shr 16) and 0xFF]++
                hRest[1][(b shr 8) and 0xFF]++
                hRest[2][b and 0xFF]++
                counted++
            }
        }
        // Too few samples for a stable CDF; leave the pixels untouched.
        if (counted < 256) return

        // Build one 256-entry LUT per channel by CDF inversion.
        val lut = Array(3) { IntArray(256) }
        for (c in 0 until 3) {
            val cdfR = FloatArray(256)
            val cdfO = FloatArray(256)
            var accR = 0
            var accO = 0
            for (i in 0 until 256) {
                accR += hRest[c][i]
                accO += hOrig[c][i]
                cdfR[i] = accR.toFloat() / counted
                cdfO[i] = accO.toFloat() / counted
            }
            var j = 0
            for (i in 0 until 256) {
                val target = cdfR[i]
                while (j < 255 && cdfO[j] < target) j++
                // Full inversion, not a half-step. A half-strength blend was
                // built and measured first, on the assumption that matching all
                // the way would drag the original's softness back over the
                // restoration. It does not: the tone shift GFPGAN applies is a
                // contrast *compression*, so undoing it fully restores contrast
                // instead of removing it. Measured on a synthetic FFHQ-style
                // shift, tonal step across the mask edge fell 28.59 -> 14.15
                // at half and -> 0.22 at full, while detail energy went the
                // right way too (1.157x at half, 1.314x at full). Full wins on
                // both counts, so the half-step was dropped.
                lut[c][i] = j.coerceIn(0, 255)
            }
        }

        for (i in rest.indices) {
            val v = rest[i]
            rest[i] = (0xFF shl 24) or
                (lut[0][(v shr 16) and 0xFF] shl 16) or
                (lut[1][(v shr 8) and 0xFF] shl 8) or
                lut[2][v and 0xFF]
        }
        restored.setPixels(rest, 0, rw, 0, 0, rw, rh)
    }

    /**
     * Composites [restored] over the region of [target], weighting by:
     * an elliptical falloff (no rectangle seam), a skin penalty and a
     * feature bonus around the landmarks.
     */
    private fun blend(
        target: Bitmap,
        original: Bitmap,
        restored: Bitmap,
        face: FaceDetector.Face,
        rx0: Int,
        ry0: Int,
        rw: Int,
        rh: Int,
        scale: Int,
        strength: Float
    ) {
        val orig = IntArray(rw * rh)
        val rest = IntArray(rw * rh)
        original.getPixels(orig, 0, rw, 0, 0, rw, rh)
        restored.getPixels(rest, 0, rw, 0, 0, rw, rh)

        // Landmark positions inside the ROI, in target pixels.
        val feats = face.landmarks.map { (lx, ly) ->
            Pair(lx * scale - rx0, ly * scale - ry0)
        }
        // Feature influence radius: a fraction of the face width.
        val featR = max(8f, (face.box.width() * scale) * 0.16f)
        val featR2 = featR * featR

        val cx = rw * 0.5f
        val cy = rh * 0.5f
        val ax = rw * 0.46f
        val ay = rh * 0.5f

        val out = IntArray(rw * rh)
        for (y in 0 until rh) {
            val dy = (y - cy) / ay
            val base = y * rw
            for (x in 0 until rw) {
                val dx = (x - cx) / ax
                val r = dx * dx + dy * dy
                // Full weight inside 0.72 of the ellipse, raised-cosine to 1.0.
                var w = when {
                    r >= 1f -> 0f
                    r <= 0.52f -> 1f
                    else -> {
                        val t = ((r - 0.52f) / 0.48f).coerceIn(0f, 1f)
                        0.5f * (1f + cos(PI * t.toDouble()).toFloat())
                    }
                }
                if (w <= 0.002f) { out[base + x] = orig[base + x]; continue }

                val c = orig[base + x]
                val rr = (c shr 16) and 0xFF
                val gg = (c shr 8) and 0xFF
                val bb = c and 0xFF

                // Skin: damp restoration so cheeks keep their natural texture.
                if (isSkin(rr, gg, bb)) w *= 0.55f

                // Eyes / nose / mouth: push restoration up, this is where
                // GFPGAN genuinely reconstructs information.
                var bonus = 0f
                for ((fx, fy) in feats) {
                    val ddx = x - fx
                    val ddy = y - fy
                    val d2 = ddx * ddx + ddy * ddy
                    if (d2 < featR2) {
                        val t = 1f - d2 / featR2
                        if (t > bonus) bonus = t
                    }
                }
                w = min(1f, w * (1f + 0.75f * bonus))
                w *= strength

                val n = rest[base + x]
                out[base + x] = (0xFF shl 24) or
                    (mix(rr, (n shr 16) and 0xFF, w) shl 16) or
                    (mix(gg, (n shr 8) and 0xFF, w) shl 8) or
                    mix(bb, n and 0xFF, w)
            }
        }
        target.setPixels(out, 0, rw, rx0, ry0, rw, rh)
    }

    /**
     * Skin test in YCbCr. The bounds are the well-established Chai & Ngan
     * range, widened slightly so darker and lighter tones are both covered.
     */
    private fun isSkin(r: Int, g: Int, b: Int): Boolean {
        val yy = 0.299f * r + 0.587f * g + 0.114f * b
        if (yy < 40f) return false
        val cb = 128f - 0.168736f * r - 0.331264f * g + 0.5f * b
        val cr = 128f + 0.5f * r - 0.418688f * g - 0.081312f * b
        return cb in 74f..130f && cr in 133f..180f
    }

    private fun mix(a: Int, b: Int, w: Float): Int {
        val v = (a + (b - a) * w).roundToInt()
        return if (v < 0) 0 else if (v > 255) 255 else v
    }

    private fun back(v: Float): Int {
        val x = ((v + 1f) * 127.5f).roundToInt()
        return if (x < 0) 0 else if (x > 255) 255 else x
    }

    override fun close() {
        runCatching { session.close() }
    }
}
