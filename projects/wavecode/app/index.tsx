
import React from 'react';
import { StyleSheet, View } from 'react-native';
import PagerView from 'react-native-pager-view';
import ChatScreen from '../screens/ChatScreen';
import CodeScreen from '../screens/CodeScreen';
import FileTreeScreen from '../screens/FileTreeScreen';

const App = () => {
  return (
    <PagerView style={styles.container} initialPage={0}>
      <View key="1">
        <ChatScreen />
      </View>
      <View key="2">
        <CodeScreen />
      </View>
      <View key="3">
        <FileTreeScreen />
      </View>
    </PagerView>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
  },
});

export default App;
