import 'package:flutter/material.dart';

void main() {
  runApp(const MainApp());
}

class MainApp extends StatelessWidget {
  const MainApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
        home: Scaffold(
          body: Padding(
              padding: EdgeInsets.symmetric(vertical: 20), child: const TopBar()),
        ),
        theme: new ThemeData(scaffoldBackgroundColor: const Color(0xFFFFFFFF)));
  }
}

class TopBar extends StatefulWidget {
  const TopBar({super.key});

  @override
  State<TopBar> createState() => TopBarState();
}

class TopBarState extends State<TopBar> {
  int selected = 0;
  List<Widget> selector = [
    new Text("전체"),
    new Text("동아리"),
    new Text("앱/게임"),
    new Text("웹/서버"),
    new Text("기타")
  ];

  @override
  Widget build(BuildContext context) {
    return Scaffold(
        appBar: AppBar(
      title: new Row(children: selector),
      shadowColor: const Color(0x000000FF),
    ));
  }
}
