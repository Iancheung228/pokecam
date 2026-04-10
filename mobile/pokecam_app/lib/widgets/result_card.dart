import 'package:flutter/material.dart';

/// Displays centering ratios and PSA pass/fail indicators.
/// [data] is the decoded JSON map from the /analyze endpoint.
class ResultCard extends StatelessWidget {
  final Map<String, dynamic> data;

  const ResultCard({super.key, required this.data});

  @override
  Widget build(BuildContext context) {
    final lrRatio = data['lr_ratio'] as String? ?? '?/?';
    final tbRatio = data['tb_ratio'] as String? ?? '?/?';
    final psa10Lr = data['psa10_lr_pass'] as bool? ?? false;
    final psa10Tb = data['psa10_tb_pass'] as bool? ?? false;
    final psa9Lr = data['psa9_lr_pass'] as bool? ?? false;
    final psa9Tb = data['psa9_tb_pass'] as bool? ?? false;
    final confidence = data['confidence'] as String? ?? 'unknown';

    return Container(
      margin: const EdgeInsets.all(16),
      padding: const EdgeInsets.all(20),
      decoration: BoxDecoration(
        color: Colors.grey[900],
        borderRadius: BorderRadius.circular(12),
      ),
      child: Column(
        mainAxisSize: MainAxisSize.min,
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          const Text(
            'Centering Result',
            style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold),
          ),
          const SizedBox(height: 16),
          _ratioRow('Left / Right', lrRatio, psa10Lr, psa9Lr),
          const SizedBox(height: 10),
          _ratioRow('Top / Bottom', tbRatio, psa10Tb, psa9Tb),
          if (confidence == 'low') ...[
            const SizedBox(height: 14),
            Row(
              children: const [
                Icon(Icons.warning_amber_rounded, color: Colors.orange, size: 16),
                SizedBox(width: 6),
                Text(
                  'Low confidence — ensure card is well-lit\nand visible against a plain background.',
                  style: TextStyle(color: Colors.orange, fontSize: 12),
                ),
              ],
            ),
          ],
        ],
      ),
    );
  }

  Widget _ratioRow(String label, String ratio, bool psa10, bool psa9) {
    final grade = psa10
        ? 'PSA 10'
        : psa9
            ? 'PSA 9'
            : 'Below PSA 9';
    final color = psa10
        ? Colors.green
        : psa9
            ? Colors.orange
            : Colors.red;
    final icon = psa10 || psa9 ? Icons.check_circle : Icons.cancel;

    return Row(
      children: [
        Expanded(
          child: Text(
            '$label:  $ratio',
            style: const TextStyle(fontSize: 16, fontFamily: 'monospace'),
          ),
        ),
        Icon(icon, color: color, size: 18),
        const SizedBox(width: 6),
        Text(grade, style: TextStyle(color: color, fontSize: 14)),
      ],
    );
  }
}
