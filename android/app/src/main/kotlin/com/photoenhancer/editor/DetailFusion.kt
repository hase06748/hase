package com.photoenhancer.editor

import android.graphics.Bitmap
import kotlin.math.abs
import kotlin.math.max
import kotlin.math.min
import kotlin.math.roundToInt

/**
 * Stage 5: detail fusion and adaptive sharpening.
 *
 * This replaces the old blanket unsharp mask. A global mask sharpens sky,
 * skin and out-of-focus background just as hard as it sharpens eyelashes,
 * which is exactly how a photo starts looking artificial. Here every pixel
 * gets its own gain, derived from three local measurements:
 *
 *  - Edge strength: sharpening is spent on structure, not on flat regions.
 *  - Local variance: a *very* busy neighbourhood is already at the limit of
 *    what the sensor recorded, so pushing it further only adds crunch.
 *  - Skin and sky detection: both are explicitly damped, because both are
 *    smooth by nature and the human eye immediately reads over-sharpened skin
 *    or a gritty sky as fake.
 *
 * Local contrast recovery runs first at a wider radius, restoring the
 * mid-frequency "presence" that tiled super-resolution tends to flatten,
 * before the fine sharpening pass adds acutance.
 *
 * The whole thing is finally blended against the input, which keeps the
 * result anchored to what the SR model actually produced.
 */
object DetailFusion {

    /**
     * @param amount 0..1 user-facing strength.
     * @param protectSkin damp sharpening on skin tones.
     * @param protectSky damp sharpening on large smooth blue/grey areas.
     */
    fun apply(
        bmp: Bitmap,
        amount: Float,
        protectSkin: Boolean = true,
        protectSky: Boolean = true
    ) {
        if (amount <= 0.01f) return
        val w = bmp.width
        val h = bmp.height
        if (w < 5 || h < 5) return

        // Stripe height chosen so the working set stays a few MB regardless of
        // how large the picture is. 3 rows of context feed the wide blur.
        val ctx = 4
        val stripe = max(64, min(h, 3_000_000 / max(w, 1)))
        var top = 0
        while (top < h) {
            val rows = min(stripe, h - top)
            val from = max(0, top - ctx)
            val to = min(h, top + rows + ctx)
            val n = to - from
            val px = IntArray(w * n)
            bmp.getPixels(px, 0, w, 0, from, w, n)

            val fine = IntArray(w * n)
            val wide = IntArray(w * n)
            blurBinomial3(px, fine, w, n)
            blurBox(fine, wide, w, n, 3)

            val outRow = IntArray(w * rows)
            val off = (top - from) * w

            for (y in 0 until rows) {
                val gy = y + (top - from)
                for (x in 0 until w) {
                    val i = gy * w + x
                    val s = px[i]
                    val sr = (s shr 16) and 0xFF
                    val sg = (s shr 8) and 0xFF
                    val sb = s and 0xFF

                    val f = fine[i]
                    val wd = wide[i]

                    // Local gradient from the blurred plane: using the blurred
                    // version makes the mask itself noise-tolerant.
                    val xm = if (x > 0) i - 1 else i
                    val xp = if (x < w - 1) i + 1 else i
                    val ym = if (gy > 0) i - w else i
                    val yp = if (gy < n - 1) i + w else i
                    val edge = lumaDiff(fine[xp], fine[xm]) + lumaDiff(fine[yp], fine[ym])

                    // Mask: rises from 2 gray levels, saturates around 10.
                    var gate = ((edge - 2f) / 8f).coerceIn(0f, 1f)

                    // Already-busy areas need no extra bite.
                    val busy = abs(lumaOf(s) - lumaOf(wd))
                    if (busy > 26f) gate *= (1f - ((busy - 26f) / 40f).coerceIn(0f, 0.8f))

                    if (protectSkin && isSkin(sr, sg, sb)) gate *= 0.42f
                    if (protectSky && isSky(sr, sg, sb, busy)) gate *= 0.30f

                    // Local contrast (wide radius) then acutance (fine radius).
                    val gLocal = amount * 0.55f * gate
                    val gFine = amount * 1.30f * gate

                    var rr = sr + ((sr - ((wd shr 16) and 0xFF)) * gLocal).roundToInt()
                    var gg2 = sg + ((sg - ((wd shr 8) and 0xFF)) * gLocal).roundToInt()
                    var bb = sb + ((sb - (wd and 0xFF)) * gLocal).roundToInt()

                    rr += ((rr - ((f shr 16) and 0xFF)) * gFine).roundToInt()
                    gg2 += ((gg2 - ((f shr 8) and 0xFF)) * gFine).roundToInt()
                    bb += ((bb - (f and 0xFF)) * gFine).roundToInt()

                    // Anchor to the input so nothing drifts far from the SR
                    // result: at most a 92% move towards the sharpened value.
                    outRow[y * w + x] = (0xFF shl 24) or
                        (anchor(sr, rr) shl 16) or
                        (anchor(sg, gg2) shl 8) or
                        anchor(sb, bb)
                }
            }
            bmp.setPixels(outRow, 0, w, 0, top, w, rows)
            top += rows
        }
    }

    private fun anchor(orig: Int, sharpened: Int): Int {
        val v = orig + ((sharpened - orig) * 0.95f).roundToInt()
        return if (v < 0) 0 else if (v > 255) 255 else v
    }

    /**
     * Professional HDR & Natural Color Grading Stage.
     * Applies subtle local tone mapping and micro-contrast adjustment 
     * to enhance dynamic range and pop colors without over-saturation.
     */
    fun applyHdrAndColor(bmp: Bitmap) {
        val w = bmp.width
        val h = bmp.height
        if (w < 5 || h < 5) return

        val stripe = max(64, min(h, 3_000_000 / max(w, 1)))
        var top = 0
        while (top < h) {
            val rows = min(stripe, h - top)
            val px = IntArray(w * rows)
            bmp.getPixels(px, 0, w, 0, top, w, rows)

            for (i in px.indices) {
                val c = px[i]
                var r = (c shr 16) and 0xFF
                var g = (c shr 8) and 0xFF
                var b = c and 0xFF

                // Convert to YUV / Luma for intelligent tone mapping
                val y = 0.299f * r + 0.587f * g + 0.114f * b
                
                // Subtle S-Curve for HDR micro-contrast (Shadow lifting & Highlight restraint)
                val normY = y / 255.0f
                // S-curve formula: smoothstep enhancement
                val enhancedY = normY * normY * (3.0f - 2.0f * normY)
                val deltaY = (enhancedY * 255.0f - y) * 0.18f // 18% strength for natural look

                // Apply delta to RGB while preserving color ratios (Vibrancy preservation)
                var fr = (r + deltaY).roundToInt()
                var fg = (g + deltaY).roundToInt()
                var fb = (b + deltaY).roundToInt()

                // Slight natural saturation boost (Vibrancy)
                val luma = 0.299f * fr + 0.587f * fg + 0.114f * fb
                fr = (luma + (fr - luma) * 1.08f).roundToInt()
                fg = (luma + (fg - luma) * 1.08f).roundToInt()
                fb = (luma + (fb - luma) * 1.08f).roundToInt()

                px[i] = (0xFF shl 24) or
                    (fr.coerceIn(0, 255) shl 16) or
                    (fg.coerceIn(0, 255) shl 8) or
                    fb.coerceIn(0, 255)
            }
            bmp.setPixels(px, 0, w, 0, top, w, rows)
            top += rows
        }
    }

    private fun lumaOf(c: Int): Float =
        0.2126f * ((c shr 16) and 0xFF) +
            0.7152f * ((c shr 8) and 0xFF) +
            0.0722f * (c and 0xFF)

    private fun lumaDiff(a: Int, b: Int): Float = abs(lumaOf(a) - lumaOf(b))

    private fun isSkin(r: Int, g: Int, b: Int): Boolean {
        val yy = 0.299f * r + 0.587f * g + 0.114f * b
        if (yy < 40f) return false
        val cb = 128f - 0.168736f * r - 0.331264f * g + 0.5f * b
        val cr = 128f + 0.5f * r - 0.418688f * g - 0.081312f * b
        return cb in 74f..130f && cr in 133f..180f
    }

    /**
     * Sky: bright, blue-dominant or near-neutral, and locally smooth. The
     * smoothness term is what stops blue clothing from being treated as sky.
     */
    private fun isSky(r: Int, g: Int, b: Int, busy: Float): Boolean {
        if (busy > 7f) return false
        val bright = (r + g + b) / 3
        if (bright < 105) return false
        val blueish = b >= g && g >= r && (b - r) > 12
        val neutral = abs(r - g) < 12 && abs(g - b) < 12 && bright > 170
        return blueish || neutral
    }

    /** Separable [1 2 1]/4 blur, clamped at the edges. */
    private fun blurBinomial3(src: IntArray, dst: IntArray, w: Int, h: Int) {
        val tmp = IntArray(w * h)
        for (y in 0 until h) {
            val base = y * w
            for (x in 0 until w) {
                val l = src[base + if (x > 0) x - 1 else 0]
                val c = src[base + x]
                val r = src[base + if (x < w - 1) x + 1 else w - 1]
                tmp[base + x] = weighted3(l, c, r)
            }
        }
        for (y in 0 until h) {
            val up = (if (y > 0) y - 1 else 0) * w
            val cu = y * w
            val dn = (if (y < h - 1) y + 1 else h - 1) * w
            for (x in 0 until w) dst[cu + x] = weighted3(tmp[up + x], tmp[cu + x], tmp[dn + x])
        }
    }

    /** Separable box blur of radius [r], clamped at the edges. */
    private fun blurBox(src: IntArray, dst: IntArray, w: Int, h: Int, r: Int) {
        val tmp = IntArray(w * h)
        val n = 2 * r + 1
        for (y in 0 until h) {
            val base = y * w
            for (x in 0 until w) {
                var sr = 0; var sg = 0; var sb = 0
                for (k in -r..r) {
                    val c = src[base + (x + k).coerceIn(0, w - 1)]
                    sr += (c shr 16) and 0xFF
                    sg += (c shr 8) and 0xFF
                    sb += c and 0xFF
                }
                tmp[base + x] = ((sr / n) shl 16) or ((sg / n) shl 8) or (sb / n)
            }
        }
        for (y in 0 until h) {
            for (x in 0 until w) {
                var sr = 0; var sg = 0; var sb = 0
                for (k in -r..r) {
                    val c = tmp[(y + k).coerceIn(0, h - 1) * w + x]
                    sr += (c shr 16) and 0xFF
                    sg += (c shr 8) and 0xFF
                    sb += c and 0xFF
                }
                dst[y * w + x] = ((sr / n) shl 16) or ((sg / n) shl 8) or (sb / n)
            }
        }
    }

    private fun weighted3(a: Int, b: Int, c: Int): Int {
        val r = (((a shr 16) and 0xFF) + 2 * ((b shr 16) and 0xFF) + ((c shr 16) and 0xFF)) shr 2
        val g = (((a shr 8) and 0xFF) + 2 * ((b shr 8) and 0xFF) + ((c shr 8) and 0xFF)) shr 2
        val bl = ((a and 0xFF) + 2 * (b and 0xFF) + (c and 0xFF)) shr 2
        return (r shl 16) or (g shl 8) or bl
    }
}
