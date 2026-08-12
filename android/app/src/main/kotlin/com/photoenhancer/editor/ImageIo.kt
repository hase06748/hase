package com.photoenhancer.editor

import android.content.ContentValues
import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.ImageDecoder
import android.graphics.Matrix
import android.net.Uri
import android.os.Build
import android.os.Environment
import android.provider.MediaStore
import androidx.exifinterface.media.ExifInterface
import java.io.ByteArrayOutputStream
import java.io.File
import java.io.FileOutputStream
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

/**
 * Handles decoding of PNG / JPEG / HEIF / HEIC / WEBP via the platform decoder
 * (ImageDecoder covers HEIF natively on API 28+), plus EXIF-correct orientation
 * and saving of results to the public Pictures directory.
 */
object ImageIo {

    data class Loaded(
        val bitmap: Bitmap,
        val displayName: String,
        val mimeType: String,
    )

    fun decode(ctx: Context, uri: Uri): Loaded {
        val resolver = ctx.contentResolver
        val mime = resolver.getType(uri) ?: "image/*"

        var bmp: Bitmap = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.P) {
            val source = ImageDecoder.createSource(resolver, uri)
            ImageDecoder.decodeBitmap(source) { decoder, _, _ ->
                decoder.allocator = ImageDecoder.ALLOCATOR_SOFTWARE
                decoder.isMutableRequired = false
                // 10-bit / HDR HEIF can reject an sRGB target; the decode must
                // still succeed, so treat the conversion as best effort.
                try {
                    decoder.setTargetColorSpace(
                        android.graphics.ColorSpace.get(android.graphics.ColorSpace.Named.SRGB)
                    )
                } catch (_: Throwable) {
                }
            }
        } else {
            resolver.openInputStream(uri).use { BitmapFactory.decodeStream(it) }
                ?: throw IllegalArgumentException("unsupported image")
        }

        if (bmp.config != Bitmap.Config.ARGB_8888) {
            bmp = bmp.copy(Bitmap.Config.ARGB_8888, false)
        }

        // ImageDecoder already applies EXIF for most formats; guard for the
        // BitmapFactory fallback path only.
        if (Build.VERSION.SDK_INT < Build.VERSION_CODES.P) {
            bmp = applyExif(ctx, uri, bmp)
        }

        return Loaded(bmp, queryName(ctx, uri), mime)
    }

    private fun applyExif(ctx: Context, uri: Uri, bmp: Bitmap): Bitmap {
        return try {
            ctx.contentResolver.openInputStream(uri).use { input ->
                if (input == null) return bmp
                val exif = ExifInterface(input)
                val o = exif.getAttributeInt(
                    ExifInterface.TAG_ORIENTATION, ExifInterface.ORIENTATION_NORMAL
                )
                val m = Matrix()
                when (o) {
                    ExifInterface.ORIENTATION_ROTATE_90 -> m.postRotate(90f)
                    ExifInterface.ORIENTATION_ROTATE_180 -> m.postRotate(180f)
                    ExifInterface.ORIENTATION_ROTATE_270 -> m.postRotate(270f)
                    ExifInterface.ORIENTATION_FLIP_HORIZONTAL -> m.postScale(-1f, 1f)
                    ExifInterface.ORIENTATION_FLIP_VERTICAL -> m.postScale(1f, -1f)
                    else -> return bmp
                }
                Bitmap.createBitmap(bmp, 0, 0, bmp.width, bmp.height, m, true)
            }
        } catch (t: Throwable) {
            bmp
        }
    }

    private fun queryName(ctx: Context, uri: Uri): String {
        return try {
            ctx.contentResolver.query(uri, arrayOf(MediaStore.MediaColumns.DISPLAY_NAME), null, null, null)
                ?.use { c ->
                    if (c.moveToFirst()) c.getString(0) else null
                } ?: uri.lastPathSegment ?: "image"
        } catch (t: Throwable) {
            "image"
        }
    }

    fun encode(bmp: Bitmap, format: String, quality: Int): ByteArray {
        val out = ByteArrayOutputStream(1 shl 20)
        val fmt = when (format.lowercase(Locale.US)) {
            "png" -> Bitmap.CompressFormat.PNG
            else -> Bitmap.CompressFormat.JPEG
        }
        bmp.compress(fmt, quality, out)
        return out.toByteArray()
    }

    /** Saves to Pictures/PhotoEnhancer and returns the human readable path. */
    fun saveToGallery(ctx: Context, bmp: Bitmap, baseName: String, format: String, quality: Int): String {
        val ext = if (format.equals("png", true)) "png" else "jpg"
        val mime = if (ext == "png") "image/png" else "image/jpeg"
        val stamp = SimpleDateFormat("yyyyMMdd_HHmmss", Locale.US).format(Date())
        val clean = baseName.substringBeforeLast('.').take(40).replace(Regex("[^A-Za-z0-9_\\-]"), "_")
        val name = "${clean}_x4_$stamp.$ext"
        val relDir = "${Environment.DIRECTORY_PICTURES}/PhotoEnhancer"

        return if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
            val values = ContentValues().apply {
                put(MediaStore.MediaColumns.DISPLAY_NAME, name)
                put(MediaStore.MediaColumns.MIME_TYPE, mime)
                put(MediaStore.MediaColumns.RELATIVE_PATH, relDir)
                put(MediaStore.MediaColumns.IS_PENDING, 1)
            }
            val resolver = ctx.contentResolver
            val uri = resolver.insert(MediaStore.Images.Media.EXTERNAL_CONTENT_URI, values)
                ?: throw IllegalStateException("cannot create media entry")
            resolver.openOutputStream(uri)?.use { os ->
                val fmt = if (ext == "png") Bitmap.CompressFormat.PNG else Bitmap.CompressFormat.JPEG
                bmp.compress(fmt, quality, os)
            } ?: throw IllegalStateException("cannot open output stream")
            values.clear()
            values.put(MediaStore.MediaColumns.IS_PENDING, 0)
            resolver.update(uri, values, null, null)
            "$relDir/$name"
        } else {
            val dir = File(Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_PICTURES), "PhotoEnhancer")
            if (!dir.exists()) dir.mkdirs()
            val f = File(dir, name)
            FileOutputStream(f).use { os ->
                val fmt = if (ext == "png") Bitmap.CompressFormat.PNG else Bitmap.CompressFormat.JPEG
                bmp.compress(fmt, quality, os)
            }
            MediaStore.Images.Media.insertImage(ctx.contentResolver, f.absolutePath, name, null)
            f.absolutePath
        }
    }

    /**
     * Encodes a JPEG preview no larger than [maxPixels], releasing the
     * intermediate scaled bitmap so repeated loads do not leak.
     */
    fun encodePreview(bmp: Bitmap, maxPixels: Long, quality: Int): ByteArray {
        val scaled = downscaleIfNeeded(bmp, maxPixels)
        return try {
            encode(scaled, "jpeg", quality)
        } finally {
            if (scaled !== bmp && !scaled.isRecycled) scaled.recycle()
        }
    }

    /** Writes bytes to cache for sharing; returns the file. */
    fun writeShareFile(ctx: Context, bytes: ByteArray, name: String): File {
        val dir = File(ctx.cacheDir, "share")
        if (!dir.exists()) dir.mkdirs()
        purgeOldShares(dir)
        val f = File(dir, name)
        FileOutputStream(f).use { it.write(bytes) }
        return f
    }

    /** Drops shared copies older than a day so the cache cannot grow forever. */
    private fun purgeOldShares(dir: File) {
        val cutoff = System.currentTimeMillis() - 24L * 60 * 60 * 1000
        dir.listFiles()?.forEach { f ->
            if (f.isFile && f.lastModified() < cutoff) f.delete()
        }
    }

    /** The size [downscaleIfNeeded] would produce, without touching pixels. */
    fun fittedSize(w: Int, h: Int, maxPixels: Long): Pair<Int, Int> {
        val px = w.toLong() * h.toLong()
        if (px <= maxPixels) return Pair(w, h)
        val ratio = Math.sqrt(maxPixels.toDouble() / px.toDouble())
        return Pair(
            (w * ratio).toInt().coerceAtLeast(16),
            (h * ratio).toInt().coerceAtLeast(16)
        )
    }

    fun downscaleIfNeeded(bmp: Bitmap, maxPixels: Long): Bitmap {
        val px = bmp.width.toLong() * bmp.height.toLong()
        if (px <= maxPixels) return bmp
        val ratio = Math.sqrt(maxPixels.toDouble() / px.toDouble())
        val nw = (bmp.width * ratio).toInt().coerceAtLeast(16)
        val nh = (bmp.height * ratio).toInt().coerceAtLeast(16)
        return Bitmap.createScaledBitmap(bmp, nw, nh, true)
    }
}
