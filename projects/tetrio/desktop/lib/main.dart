import 'dart:convert';

import 'package:flutter/material.dart';
import "package:http/http.dart" as http;

void main() {
  runApp(const MainApp());
}

class MainApp extends StatelessWidget {
  const MainApp({super.key});

  Future<Widget> tetriostatf() async {
    Map<String, dynamic> tetriostat = jsonDecode(await http.read(Uri.parse("https://ch.tetr.io/api/general/stats")));
    Map<String, dynamic> data = tetriostat["data"];
    if (tetriostat["success"] == true) {
      return Text("""
          Currently ${data['usercount']} registered users and ${data['anoncount']} anon users playing
          Ranked acoount is ${data['rankedcount']}, replay count is ${data['replaycount']}
        """, style: const TextStyle(fontSize:20), textAlign: TextAlign.center);
    } else {
      return const Text("error");
    }
  }

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      home: Scaffold(
        body: Center(
          child: ListView(
            children: <Widget>[
              FutureBuilder(
                future: tetriostatf(),
                builder: (context, snapshot) {
                  if (!snapshot.hasData) {
                    return const Text("loading");
                  } else if (snapshot.hasError) {
                    return const Text("error");
                  } else {
                    return snapshot.data!;
                  }
                }
              )
            ]
          ),
        ),
      ),
    );
  }
}
