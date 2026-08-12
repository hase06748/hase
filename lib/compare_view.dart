import 'dart:typed_data';
import 'dart:ui' as ui;

import 'package:flutter/material.dart';

import 'theme.dart';

/// Interactive before/after viewer.
///
/// The two images are drawn at the *same* on-screen rect, the "after" image
/// clipped to a draggable divider. Pinch/pan zoom is shared by both layers so
/// the comparison stays pixel-aligned at any magnification.
class CompareView extends StatefulWidget {
  final Uint8List before;
  final Uint8List? after;
  final double aspectRatio;

  const CompareView({
    super.key,
    required this.before,
    required this.after,
    required this.aspectRatio,
  });

  @override
  State<CompareView> createState() => _CompareViewState();
}

class _CompareViewState extends State<CompareView> {
  final TransformationController _tc = TransformationController();
  double _split = 0.5;
  ui.Image? _beforeImg;
  ui.Image? _afterImg;

  @override
  void initState() {
    super.initState();
    _decodeAll();
  }

  @override
  void didUpdateWidget(covariant CompareView old) {
    super.didUpdateWidget(old);
    if (old.before != widget.before || old.after != widget.after) {
      _decodeAll();
    }
  }

  Future<void> _decodeAll() async {
    final b = await _decode(widget.before);
    ui.Image? a;
    if (widget.after != null) a = await _decode(widget.after!);
    if (!mounted) return;
    setState(() {
      _beforeImg?.dispose();
      _afterImg?.dispose();
      _beforeImg = b;
      _afterImg = a;
    });
  }

  Future<ui.Image> _decode(Uint8List bytes) async {
    final codec = await ui.instantiateImageCodec(bytes);
    final frame = await codec.getNextFrame();
    return frame.image;
  }

  @override
  void dispose() {
    _beforeImg?.dispose();
    _afterImg?.dispose();
    _tc.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final before = _beforeImg;
    if (before == null) {
      return const Center(
        child: SizedBox(
          width: 26,
          height: 26,
          child: CircularProgressIndicator(strokeWidth: 2.4),
        ),
      );
    }

    return LayoutBuilder(
      builder: (context, c) {
        return ClipRRect(
          borderRadius: BorderRadius.circular(16),
          child: Stack(
            children: [
              Positioned.fill(
                child: InteractiveViewer(
                  transformationController: _tc,
                  minScale: 1,
                  maxScale: 8,
                  clipBehavior: Clip.hardEdge,
                  child: AspectRatio(
                    aspectRatio: widget.aspectRatio,
                    child: CustomPaint(
                      painter: _ComparePainter(
                        before: before,
                        after: _afterImg,
                        split: _split,
                      ),
                      size: Size.infinite,
                    ),
                  ),
                ),
              ),
              if (_afterImg != null) ..._buildHandle(c.maxWidth),
              if (_afterImg != null)
                Positioned(
                  left: 12,
                  top: 12,
                  child: _tag('الأصلية', AppTheme.textMid),
                ),
              if (_afterImg != null)
                Positioned(
                  right: 12,
                  top: 12,
                  child: _tag('المحسّنة ×4', AppTheme.accent),
                ),
            ],
          ),
        );
      },
    );
  }

  List<Widget> _buildHandle(double w) {
    return [
      Positioned.fill(
        child: GestureDetector(
          behavior: HitTestBehavior.translucent,
          onHorizontalDragUpdate: (d) {
            setState(() {
              _split = (_split + d.delta.dx / w).clamp(0.0, 1.0);
            });
          },
        ),
      ),
      Positioned(
        left: _split * w - 1,
        top: 0,
        bottom: 0,
        child: IgnorePointer(
          child: Container(width: 2, color: Colors.white.withValues(alpha: 0.85)),
        ),
      ),
      Positioned(
        left: _split * w - 18,
        top: 0,
        bottom: 0,
        child: IgnorePointer(
          child: Center(
            child: Container(
              width: 36,
              height: 36,
              decoration: BoxDecoration(
                color: Colors.white,
                shape: BoxShape.circle,
                boxShadow: [
                  BoxShadow(
                    color: Colors.black.withValues(alpha: 0.4),
                    blurRadius: 8,
                  ),
                ],
              ),
              child: const Icon(Icons.code, size: 20, color: Color(0xFF10131A)),
            ),
          ),
        ),
      ),
    ];
  }

  Widget _tag(String text, Color color) => Container(
        padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 5),
        decoration: BoxDecoration(
          color: Colors.black.withValues(alpha: 0.55),
          borderRadius: BorderRadius.circular(8),
        ),
        child: Text(
          text,
          style: TextStyle(
            color: color,
            fontSize: 11.5,
            fontWeight: FontWeight.w700,
          ),
        ),
      );
}

class _ComparePainter extends CustomPainter {
  final ui.Image before;
  final ui.Image? after;
  final double split;

  _ComparePainter({required this.before, required this.after, required this.split});

  @override
  void paint(Canvas canvas, Size size) {
    final dst = Offset.zero & size;
    final paint = Paint()
      ..filterQuality = FilterQuality.high
      ..isAntiAlias = true;

    // "Before" is upscaled with plain bilinear so the AI gain is visible.
    canvas.drawImageRect(
      before,
      Rect.fromLTWH(0, 0, before.width.toDouble(), before.height.toDouble()),
      dst,
      paint,
    );

    final a = after;
    if (a != null) {
      canvas.save();
      canvas.clipRect(Rect.fromLTWH(size.width * split, 0,
          size.width * (1 - split), size.height));
      canvas.drawImageRect(
        a,
        Rect.fromLTWH(0, 0, a.width.toDouble(), a.height.toDouble()),
        dst,
        paint,
      );
      canvas.restore();
    }
  }

  @override
  bool shouldRepaint(covariant _ComparePainter old) =>
      old.before != before || old.after != after || old.split != split;
}
