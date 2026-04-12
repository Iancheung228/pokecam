import 'dart:convert';
import 'package:flutter/material.dart';

class ResultCard extends StatelessWidget {
  final Map<String, dynamic> data;

  const ResultCard({super.key, required this.data});

  @override
  Widget build(BuildContext context) {
    final lrRatio   = data['lr_ratio']      as String? ?? '?/?';
    final tbRatio   = data['tb_ratio']      as String? ?? '?/?';
    final psa10Lr   = data['psa10_lr_pass'] as bool?   ?? false;
    final psa10Tb   = data['psa10_tb_pass'] as bool?   ?? false;
    final psa9Lr    = data['psa9_lr_pass']  as bool?   ?? false;
    final psa9Tb    = data['psa9_tb_pass']  as bool?   ?? false;
    final confidence = data['confidence']   as String? ?? 'unknown';
    final debugB64  = data['debug_image_b64'] as String?;

    final bool psa10 = psa10Lr && psa10Tb;
    final bool psa9  = psa9Lr  && psa9Tb;

    final String gradeLabel;
    final Color  gradeColor;
    final IconData gradeIcon;
    if (confidence == 'no_inner_border') {
      gradeLabel = 'Full Art';
      gradeColor = const Color(0xFF888888);
      gradeIcon  = Icons.help_outline;
    } else if (psa10) {
      gradeLabel = 'PSA 10';
      gradeColor = const Color(0xFF34C759);
      gradeIcon  = Icons.verified;
    } else if (psa9) {
      gradeLabel = 'PSA 9';
      gradeColor = const Color(0xFFFF9F0A);
      gradeIcon  = Icons.check_circle_outline;
    } else {
      gradeLabel = 'Below PSA 9';
      gradeColor = const Color(0xFFFF453A);
      gradeIcon  = Icons.cancel_outlined;
    }

    return Column(
      crossAxisAlignment: CrossAxisAlignment.stretch,
      children: [
        // ── Debug image (pinch-zoomable) ──────────────────────────────────
        if (debugB64 != null)
          Container(
            color: Colors.black,
            child: InteractiveViewer(
              minScale: 0.8,
              maxScale: 5.0,
              child: Image.memory(
                base64Decode(debugB64),
                fit: BoxFit.contain,
              ),
            ),
          ),

        // ── Grade badge ───────────────────────────────────────────────────
        Container(
          margin: const EdgeInsets.fromLTRB(16, 16, 16, 0),
          padding: const EdgeInsets.symmetric(vertical: 20, horizontal: 24),
          decoration: BoxDecoration(
            color: gradeColor.withOpacity(0.1),
            borderRadius: const BorderRadius.vertical(top: Radius.circular(16)),
            border: Border.all(color: gradeColor.withOpacity(0.35), width: 1.5),
          ),
          child: Row(
            children: [
              Icon(gradeIcon, color: gradeColor, size: 36),
              const SizedBox(width: 16),
              Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text(
                    gradeLabel,
                    style: TextStyle(
                      color: gradeColor,
                      fontSize: 26,
                      fontWeight: FontWeight.w700,
                    ),
                  ),
                  Text(
                    psa10
                        ? 'Passes PSA 10 centering standards'
                        : psa9
                            ? 'Passes PSA 9 centering standards'
                            : confidence == 'no_inner_border'
                                ? 'Full-art card — centering not measurable'
                                : 'Does not meet PSA 9 standards',
                    style: const TextStyle(
                      color: Color(0xFF888888),
                      fontSize: 13,
                    ),
                  ),
                ],
              ),
            ],
          ),
        ),

        // ── Measurements ──────────────────────────────────────────────────
        Container(
          margin: const EdgeInsets.fromLTRB(16, 0, 16, 16),
          padding: const EdgeInsets.all(20),
          decoration: BoxDecoration(
            color: const Color(0xFF1C1C1E),
            borderRadius: const BorderRadius.vertical(bottom: Radius.circular(16)),
            border: Border(
              left: BorderSide(color: gradeColor.withOpacity(0.35), width: 1.5),
              right: BorderSide(color: gradeColor.withOpacity(0.35), width: 1.5),
              bottom: BorderSide(color: gradeColor.withOpacity(0.35), width: 1.5),
            ),
          ),
          child: Column(
            children: [
              _MeasurementRow(
                label: 'Left / Right',
                ratio: lrRatio,
                psa10: psa10Lr,
                psa9: psa9Lr,
              ),
              const Divider(color: Color(0xFF2C2C2E), height: 24),
              _MeasurementRow(
                label: 'Top / Bottom',
                ratio: tbRatio,
                psa10: psa10Tb,
                psa9: psa9Tb,
              ),
              if (confidence == 'low') ...[
                const SizedBox(height: 16),
                Container(
                  padding: const EdgeInsets.all(12),
                  decoration: BoxDecoration(
                    color: Colors.orange.withOpacity(0.1),
                    borderRadius: BorderRadius.circular(10),
                    border: Border.all(color: Colors.orange.withOpacity(0.3)),
                  ),
                  child: const Row(
                    children: [
                      Icon(Icons.warning_amber_rounded, color: Colors.orange, size: 18),
                      SizedBox(width: 10),
                      Expanded(
                        child: Text(
                          'Low confidence — ensure the card is well-lit and flat against a plain background.',
                          style: TextStyle(color: Colors.orange, fontSize: 12, height: 1.4),
                        ),
                      ),
                    ],
                  ),
                ),
              ],
            ],
          ),
        ),
      ],
    );
  }
}

class _MeasurementRow extends StatelessWidget {
  final String label;
  final String ratio;
  final bool psa10;
  final bool psa9;

  const _MeasurementRow({
    required this.label,
    required this.ratio,
    required this.psa10,
    required this.psa9,
  });

  @override
  Widget build(BuildContext context) {
    final grade = psa10 ? 'PSA 10' : psa9 ? 'PSA 9' : 'Below 9';
    final color = psa10
        ? const Color(0xFF34C759)
        : psa9
            ? const Color(0xFFFF9F0A)
            : const Color(0xFFFF453A);

    return Row(
      children: [
        // Label
        SizedBox(
          width: 90,
          child: Text(
            label,
            style: const TextStyle(color: Color(0xFF888888), fontSize: 13),
          ),
        ),
        // Ratio (monospace, large)
        Expanded(
          child: Text(
            ratio,
            style: const TextStyle(
              color: Colors.white,
              fontSize: 22,
              fontWeight: FontWeight.w600,
              fontFamily: 'monospace',
            ),
          ),
        ),
        // Grade chip
        Container(
          padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 4),
          decoration: BoxDecoration(
            color: color.withOpacity(0.15),
            borderRadius: BorderRadius.circular(20),
            border: Border.all(color: color.withOpacity(0.4)),
          ),
          child: Text(
            grade,
            style: TextStyle(
              color: color,
              fontSize: 12,
              fontWeight: FontWeight.w600,
            ),
          ),
        ),
      ],
    );
  }
}
