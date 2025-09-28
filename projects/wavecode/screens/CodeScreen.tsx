
import React, { useState, useMemo } from 'react';
import { View, Text, StyleSheet, ScrollView, Pressable, Modal, TextInput, Alert } from 'react-native';
import { useProjectStore } from '../state/projectStore';
import { WebView } from 'react-native-webview';
import { SafeAreaView } from 'react-native-safe-area-context';

const CodeScreen = () => {
  const { files, activeFile, updateFile } = useProjectStore();
  const [modalVisible, setModalVisible] = useState(false);
  const [editingColor, setEditingColor] = useState<{ value: string; index: number } | null>(null);
  const [newColor, setNewColor] = useState('');

  const activeCode = activeFile ? files[activeFile] : 'No file selected';

  const getPreviewHtml = () => {
    const html = files['index.html'] || '';
    const css = files['style.css'] || '';
    return `<style>${css}</style>${html}`;
  };

  const handleColorPress = (color: string, index: number) => {
    setEditingColor({ value: color, index });
    setNewColor(color);
  };

  const handleColorUpdate = () => {
    if (editingColor && activeFile) {
      const newCode = activeCode.substring(0, editingColor.index) + newColor + activeCode.substring(editingColor.index + editingColor.value.length);
      updateFile(activeFile, newCode);
      setEditingColor(null);
    }
  };

  const codeElements = useMemo(() => {
    const colorRegex = /(#[0-9a-fA-F]{3,6})/g;
    let lastIndex = 0;
    const elements = [];
    let match;

    while ((match = colorRegex.exec(activeCode)) !== null) {
      if (match.index > lastIndex) {
        elements.push(<Text key={lastIndex}>{activeCode.substring(lastIndex, match.index)}</Text>);
      }
      const color = match[0];
      const index = match.index;
      elements.push(
        <Pressable key={index} onPress={() => handleColorPress(color, index)}>
          <Text style={{ color: color, backgroundColor: '#e0e0e0' }}>{color}</Text>
        </Pressable>
      );
      lastIndex = colorRegex.lastIndex;
    }

    if (lastIndex < activeCode.length) {
      elements.push(<Text key={lastIndex}>{activeCode.substring(lastIndex)}</Text>);
    }

    return elements;
  }, [activeCode]);

  return (
    <SafeAreaView style={styles.container}>
      <View style={styles.header}>
        <Text style={styles.filename}>{activeFile || 'No File Selected'}</Text>
      </View>
      <ScrollView style={styles.codeContainer}>
        <Text style={styles.codeText}>{codeElements}</Text>
      </ScrollView>
      <Pressable style={styles.previewButton} onPress={() => setModalVisible(true)}>
        <Text style={styles.previewButtonText}>Preview</Text>
      </Pressable>

      {/* Preview Modal */}
      <Modal visible={modalVisible} onRequestClose={() => setModalVisible(false)} animationType="slide">
        <WebView source={{ html: getPreviewHtml() }} style={{ flex: 1, marginTop: 40 }} />
        <Pressable style={[styles.previewButton, styles.closeButton]} onPress={() => setModalVisible(false)}>
          <Text style={styles.previewButtonText}>Close</Text>
        </Pressable>
      </Modal>

      {/* Color Editing Modal */}
      <Modal visible={!!editingColor} onRequestClose={() => setEditingColor(null)} transparent>
        <View style={styles.modalContainer}>
          <View style={styles.modalContent}>
            <Text style={styles.modalTitle}>Edit Color</Text>
            <TextInput
              style={styles.input}
              value={newColor}
              onChangeText={setNewColor}
              autoCapitalize="none"
              autoFocus
            />
            <View style={styles.buttonContainer}>
              <Pressable style={styles.modalButton} onPress={handleColorUpdate}>
                <Text style={styles.previewButtonText}>Update</Text>
              </Pressable>
              <Pressable style={[styles.modalButton, styles.closeButton]} onPress={() => setEditingColor(null)}>
                <Text style={styles.previewButtonText}>Cancel</Text>
              </Pressable>
            </View>
          </View>
        </View>
      </Modal>
    </SafeAreaView>
  );
};

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: '#f5f5f5' },
  header: { padding: 15, backgroundColor: '#e0e0e0', alignItems: 'center' },
  filename: { fontSize: 16, fontWeight: 'bold' },
  codeContainer: { flex: 1, backgroundColor: '#fff', padding: 15 },
  codeText: { fontFamily: 'monospace', fontSize: 14, color: '#333' },
  previewButton: { backgroundColor: '#6200ee', padding: 15, alignItems: 'center', justifyContent: 'center', margin: 15, borderRadius: 8 },
  previewButtonText: { color: '#fff', fontSize: 16, fontWeight: 'bold' },
  closeButton: { backgroundColor: '#c62828' },
  modalContainer: { flex: 1, justifyContent: 'center', alignItems: 'center', backgroundColor: 'rgba(0,0,0,0.5)' },
  modalContent: { backgroundColor: '#fff', padding: 20, borderRadius: 10, width: '80%' },
  modalTitle: { fontSize: 20, fontWeight: 'bold', marginBottom: 15 },
  input: { borderWidth: 1, borderColor: '#ccc', padding: 10, borderRadius: 5, marginBottom: 15 },
  buttonContainer: { flexDirection: 'row', justifyContent: 'space-around' },
  modalButton: { backgroundColor: '#6200ee', padding: 10, borderRadius: 5, flex: 0.45, alignItems: 'center' },
});

export default CodeScreen;
