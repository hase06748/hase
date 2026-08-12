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
import kotlin.math.exp
import kotlin.math.max
import kotlin.math.min
import kotlin.math.roundToInt

/**
 * YuNet face detector (228 KB) used by stage 1 to answer "are there faces?"
 * and by stage 4 to know exactly where they are.
 *
 * The decoding here was validated offline against a NumPy reference before
 * being written: letterbox into a fixed 640x640 canvas, read three strides,
 * combine the classification and objectness heads as sqrt(cls * obj), decode
 * centre/size in stride units, then greedy NMS. Feeding the model BGR (which
 * is what it was trained on) measurably raises the scores, so we do that.
 */
class FaceDetector private constructor(
    private val env: OrtEnvironment,
    private val session: OrtSession
) : Closeable {

    companion object {
        private const val ASSET = "yunet.onnx"
        private const val SIZE = 640
        private val STRIDES = intArrayOf(8, 16, 32)

        /** Returns null when the model cannot be loaded; callers degrade. */
        fun create(ctx: Context, env: OrtEnvironment): FaceDetector? = try {
            val path = ModelStore.ensure(ctx, ASSET)
            // A detector only needs box coordinates to be roughly right, and
            // every candidate is filtered by a confidence threshold and NMS
            // afterwards, so fp16 is safe here.
            val s = Accelerator.create(
                env = env,
                modelPath = path,
                label = "كشف",
                threads = 2,
                precision = Accelerator.Precision.RELAXED
            )
            if (s == null) null else FaceDetector(env, s)
        } catch (t: Throwable) {
            null
        }
    }

    /**
     * @param box  face rectangle in source pixels.
     * @param score confidence 0..1.
     * @param landmarks five points (eyes, nose, mouth corners) in source pixels.
     */
    data class Face(
        val box: Rect,
        val score: Float,
        val landmarks: List<Pair<Float, Float>>
    ) {
        val cx: Float get() = box.exactCenterX()
        val cy: Float get() = box.exactCenterY()
    }

    fun detect(src: Bitmap, confThreshold: Float = 0.6f, nmsThreshold: Float = 0.3f): List<Face> {
        val ow = src.width
        val oh = src.height
        val ratio = min(SIZE.toFloat() / ow, SIZE.toFloat() / oh)
        val nw = max(1, (ow * ratio).roundToInt())
        val nh = max(1, (oh * ratio).roundToInt())

        // Letterbox top-left; the padding stays black and decodes to nothing.
        val canvasBmp = Bitmap.createBitmap(SIZE, SIZE, Bitmap.Config.ARGB_8888)
        try {
            Canvas(canvasBmp).apply {
                drawColor(0xFF000000.toInt())
                drawBitmap(src, null, Rect(0, 0, nw, nh), Paint(Paint.FILTER_BITMAP_FLAG))
            }

            val buf = FloatBuffer.allocate(3 * SIZE * SIZE)
            val arr = buf.array()
            val plane = SIZE * SIZE
            val row = IntArray(SIZE)
            for (y in 0 until SIZE) {
                canvasBmp.getPixels(row, 0, SIZE, 0, y, SIZE, 1)
                val base = y * SIZE
                for (x in 0 until SIZE) {
                    val c = row[x]
                    // BGR order, raw 0..255 (the model normalises internally).
                    arr[base + x] = (c and 0xFF).toFloat()
                    arr[plane + base + x] = ((c shr 8) and 0xFF).toFloat()
                    arr[2 * plane + base + x] = ((c shr 16) and 0xFF).toFloat()
                }
            }
            buf.rewind()

            val inName = session.inputNames.iterator().next()
            val shape = longArrayOf(1, 3, SIZE.toLong(), SIZE.toLong())
            val raw = HashMap<String, FloatArray>()
            OnnxTensor.createTensor(env, buf, shape).use { tensor ->
                session.run(mapOf(inName to tensor)).use { res ->
                    for (i in 0 until res.size()) {
                        val name = session.outputNames.toList()[i]
                        val fb = (res[i] as OnnxTensor).floatBuffer
                        val a = FloatArray(fb.capacity())
                        fb.get(a)
                        raw[name] = a
                    }
                }
            }

            val boxes = ArrayList<FloatArray>()   // x, y, w, h in canvas pixels
            val scores = ArrayList<Float>()
            val kps = ArrayList<FloatArray>()

            for (s in STRIDES) {
                val cls = raw["cls_$s"] ?: continue
                val obj = raw["obj_$s"] ?: continue
                val bbox = raw["bbox_$s"] ?: continue
                val kp = raw["kps_$s"]
                val cols = SIZE / s
                for (i in cls.indices) {
                    if (i >= obj.size) break
                    val c = cls[i].coerceIn(0f, 1f)
                    val o = obj[i].coerceIn(0f, 1f)
                    val sc = Math.sqrt((c * o).toDouble()).toFloat()
                    if (sc <= confThreshold) continue
                    val b = i * 4
                    if (b + 3 >= bbox.size) break
                    val col = i % cols
                    val rowI = i / cols
                    val cx = (col + bbox[b]) * s
                    val cy = (rowI + bbox[b + 1]) * s
                    val w = exp(bbox[b + 2].toDouble()).toFloat() * s
                    val h = exp(bbox[b + 3].toDouble()).toFloat() * s
                    boxes.add(floatArrayOf(cx - w / 2f, cy - h / 2f, w, h))
                    scores.add(sc)
                    val pts = FloatArray(10)
                    if (kp != null && i * 10 + 9 < kp.size) {
                        for (k in 0 until 5) {
                            pts[2 * k] = (col + kp[i * 10 + 2 * k]) * s
                            pts[2 * k + 1] = (rowI + kp[i * 10 + 2 * k + 1]) * s
                        }
                    }
                    kps.add(pts)
                }
            }

            if (boxes.isEmpty()) return emptyList()

            val order = scores.indices.sortedByDescending { scores[it] }.toMutableList()
            val keep = ArrayList<Int>()
            while (order.isNotEmpty()) {
                val i = order.removeAt(0)
                keep.add(i)
                order.removeAll { j -> iou(boxes[i], boxes[j]) > nmsThreshold }
            }

            val inv = 1f / ratio
            return keep.map { i ->
                val b = boxes[i]
                val x0 = (b[0] * inv).roundToInt().coerceIn(0, ow - 1)
                val y0 = (b[1] * inv).roundToInt().coerceIn(0, oh - 1)
                val x1 = ((b[0] + b[2]) * inv).roundToInt().coerceIn(x0 + 1, ow)
                val y1 = ((b[1] + b[3]) * inv).roundToInt().coerceIn(y0 + 1, oh)
                val pts = ArrayList<Pair<Float, Float>>(5)
                val k = kps[i]
                for (n in 0 until 5) pts.add(Pair(k[2 * n] * inv, k[2 * n + 1] * inv))
                Face(Rect(x0, y0, x1, y1), scores[i], pts)
            }
        } finally {
            canvasBmp.recycle()
        }
    }

    private fun iou(a: FloatArray, b: FloatArray): Float {
        val x1 = max(a[0], b[0])
        val y1 = max(a[1], b[1])
        val x2 = min(a[0] + a[2], b[0] + b[2])
        val y2 = min(a[1] + a[3], b[1] + b[3])
        val iw = x2 - x1
        val ih = y2 - y1
        if (iw <= 0f || ih <= 0f) return 0f
        val inter = iw * ih
        return inter / (a[2] * a[3] + b[2] * b[3] - inter)
    }

    override fun close() {
        runCatching { session.close() }
    }
}
