import 'dart:convert';
import 'package:cross_file/cross_file.dart';
import 'package:http/http.dart' as http;
import 'package:http_parser/http_parser.dart';

class ApiService {
  // Dev: replace with your machine's WiFi IP, e.g. 'http://192.168.1.42:8080'
  // Prod: replace with your Cloud Run URL
  static const String baseUrl = 'https://pokecam-api-633662296510.us-central1.run.app';

  /// Sends [imageFile] to the /analyze endpoint and returns the decoded JSON map.
  /// Throws an [ApiException] on non-200 responses.
  static Future<Map<String, dynamic>> analyzeCard(XFile imageFile) async {
    final uri = Uri.parse('$baseUrl/analyze');
    final bytes = await imageFile.readAsBytes();

    final request = http.MultipartRequest('POST', uri)
      ..files.add(http.MultipartFile.fromBytes(
        'file',
        bytes,
        filename: 'card.jpg',
        contentType: MediaType('image', 'jpeg'),
      ));

    final streamedResponse =
        await request.send().timeout(const Duration(seconds: 20));
    final body = await streamedResponse.stream.bytesToString();

    if (streamedResponse.statusCode == 200) {
      return jsonDecode(body) as Map<String, dynamic>;
    }

    // Surface server error message to the UI
    String detail = streamedResponse.statusCode.toString();
    try {
      final decoded = jsonDecode(body) as Map<String, dynamic>;
      detail = decoded['detail']?.toString() ?? detail;
    } catch (_) {}

    throw ApiException(streamedResponse.statusCode, detail);
  }
}

class ApiException implements Exception {
  final int statusCode;
  final String message;
  const ApiException(this.statusCode, this.message);

  @override
  String toString() => 'ApiException($statusCode): $message';
}
