import 'package:flutter/material.dart';
import 'screens/camera_screen.dart';

void main() {
  runApp(const PokecamApp());
}

class PokecamApp extends StatelessWidget {
  const PokecamApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Pokecam',
      theme: ThemeData.dark(useMaterial3: true),
      home: const CameraScreen(),
      debugShowCheckedModeBanner: false,
    );
  }
}
