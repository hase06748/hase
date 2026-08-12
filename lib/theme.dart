import 'package:flutter/material.dart';

/// Dark, image-first visual language: neutral surfaces so photos read
/// accurately, with a single cyan accent for actions.
class AppTheme {
  static const Color bg = Color(0xFF0C0E12);
  static const Color surface = Color(0xFF15181F);
  static const Color surfaceHigh = Color(0xFF1E222B);
  static const Color outline = Color(0xFF2A2F3A);
  static const Color accent = Color(0xFF35D6C4);
  static const Color accentDim = Color(0xFF1F8C82);
  static const Color textHi = Color(0xFFF1F3F7);
  static const Color textMid = Color(0xFF9BA3B4);
  static const Color danger = Color(0xFFE5556E);

  static ThemeData build() {
    final base = ThemeData.dark(useMaterial3: true);
    return base.copyWith(
      scaffoldBackgroundColor: bg,
      colorScheme: base.colorScheme.copyWith(
        surface: surface,
        primary: accent,
        secondary: accent,
        error: danger,
        onPrimary: const Color(0xFF04140F),
      ),
      appBarTheme: const AppBarTheme(
        backgroundColor: bg,
        surfaceTintColor: Colors.transparent,
        elevation: 0,
        centerTitle: false,
        titleTextStyle: TextStyle(
          color: textHi,
          fontSize: 19,
          fontWeight: FontWeight.w600,
          letterSpacing: 0.1,
        ),
      ),
      cardTheme: CardThemeData(
        color: surface,
        elevation: 0,
        shape: RoundedRectangleBorder(
          borderRadius: BorderRadius.circular(18),
          side: const BorderSide(color: outline),
        ),
        margin: EdgeInsets.zero,
      ),
      dialogTheme: DialogThemeData(
        backgroundColor: surfaceHigh,
        surfaceTintColor: Colors.transparent,
        shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(20)),
      ),
      snackBarTheme: SnackBarThemeData(
        backgroundColor: surfaceHigh,
        contentTextStyle: const TextStyle(color: textHi),
        behavior: SnackBarBehavior.floating,
        shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
      ),
      filledButtonTheme: FilledButtonThemeData(
        style: FilledButton.styleFrom(
          backgroundColor: accent,
          foregroundColor: const Color(0xFF04140F),
          minimumSize: const Size(0, 52),
          textStyle: const TextStyle(fontSize: 15.5, fontWeight: FontWeight.w700),
          shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(14)),
        ),
      ),
      outlinedButtonTheme: OutlinedButtonThemeData(
        style: OutlinedButton.styleFrom(
          foregroundColor: textHi,
          minimumSize: const Size(0, 50),
          side: const BorderSide(color: outline),
          textStyle: const TextStyle(fontSize: 15, fontWeight: FontWeight.w600),
          shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(14)),
        ),
      ),
      textButtonTheme: TextButtonThemeData(
        style: TextButton.styleFrom(foregroundColor: accent),
      ),
      sliderTheme: SliderThemeData(
        activeTrackColor: accent,
        inactiveTrackColor: outline,
        thumbColor: accent,
        overlayColor: accent.withValues(alpha: 0.14),
        trackHeight: 3,
      ),
      dividerTheme: const DividerThemeData(color: outline, thickness: 1, space: 1),
      progressIndicatorTheme: const ProgressIndicatorThemeData(
        color: accent,
        linearTrackColor: outline,
      ),
      textTheme: base.textTheme.apply(
        bodyColor: textHi,
        displayColor: textHi,
      ),
    );
  }
}
