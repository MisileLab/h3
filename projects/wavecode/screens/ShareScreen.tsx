
import React, { useState } from 'react';
import { View, Text, StyleSheet, Pressable, TextInput, ScrollView, Alert } from 'react-native';
import { useProjectStore } from '../state/projectStore';
import { useRouter } from 'expo-router';

const ShareScreen = () => {
  const { exportProject, importProject } = useProjectStore();
  const router = useRouter();
  const [exportedJson, setExportedJson] = useState('');
  const [importJson, setImportJson] = useState('');

  const handleExport = () => {
    setExportedJson(exportProject());
  };

  const handleImport = () => {
    try {
      importProject(importJson);
      Alert.alert('Success', 'Project imported successfully!');
      router.back();
    } catch (error) {
      Alert.alert('Error', 'Invalid project data.');
    }
  };

  return (
    <View style={styles.container}>
      <Text style={styles.title}>Share Project</Text>

      <ScrollView style={styles.section}>
        <Text style={styles.label}>Export Project (JSON)</Text>
        <TextInput
          style={styles.jsonOutput}
          multiline
          editable={false}
          value={exportedJson}
        />
        <Pressable style={styles.button} onPress={handleExport}>
          <Text style={styles.buttonText}>Generate Export JSON</Text>
        </Pressable>
      </ScrollView>

      <View style={styles.section}>
        <Text style={styles.label}>Import Project (JSON)</Text>
        <TextInput
          style={styles.jsonInput}
          multiline
          placeholder="Paste project JSON here..."
          value={importJson}
          onChangeText={setImportJson}
        />
        <Pressable style={styles.button} onPress={handleImport}>
          <Text style={styles.buttonText}>Import Project</Text>
        </Pressable>
      </View>

      <Pressable style={styles.closeButton} onPress={() => router.back()}>
        <Text style={styles.closeButtonText}>Close</Text>
      </Pressable>
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    padding: 20,
    paddingTop: 60,
    backgroundColor: '#1e1e1e',
  },
  title: {
    fontSize: 28,
    fontWeight: 'bold',
    color: '#fff',
    marginBottom: 30,
  },
  section: {
    marginBottom: 20,
  },
  label: {
    fontSize: 18,
    color: '#fff',
    marginBottom: 10,
  },
  jsonOutput: {
    backgroundColor: '#2d2d2d',
    color: '#fff',
    padding: 10,
    borderRadius: 8,
    height: 150,
    fontFamily: 'monospace',
    marginBottom: 10,
  },
  jsonInput: {
    backgroundColor: '#2d2d2d',
    color: '#fff',
    padding: 10,
    borderRadius: 8,
    height: 150,
    fontFamily: 'monospace',
    marginBottom: 10,
  },
  button: {
    backgroundColor: '#007aff',
    padding: 15,
    borderRadius: 8,
    alignItems: 'center',
    marginBottom: 10,
  },
  buttonText: {
    color: '#fff',
    fontSize: 16,
    fontWeight: 'bold',
  },
  closeButton: {
    backgroundColor: '#6200ee',
    padding: 15,
    borderRadius: 8,
    alignItems: 'center',
    marginTop: 20,
  },
  closeButtonText: {
    color: '#fff',
    fontSize: 16,
    fontWeight: 'bold',
  },
});

export default ShareScreen;
