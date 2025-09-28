
import React, { useState } from 'react';
import { View, Text, StyleSheet, FlatList, Pressable, TextInput } from 'react-native';
import { useProjectStore } from '../state/projectStore';
import { useRouter } from 'expo-router';

const FileTreeScreen = () => {
  const { files, activeFile, setActiveFile, addFile } = useProjectStore();
  const [newFileName, setNewFileName] = useState('');
  const router = useRouter();

  const handleAddFile = () => {
    if (newFileName.trim()) {
      addFile(newFileName.trim());
      setNewFileName('');
    }
  };

  return (
    <View style={styles.container}>
      <Text style={styles.header}>File Explorer</Text>
      <FlatList
        data={Object.keys(files)}
        keyExtractor={(item) => item}
        renderItem={({ item }) => (
          <Pressable
            style={[styles.fileItem, activeFile === item && styles.activeFileItem]}
            onPress={() => setActiveFile(item)}>
            <Text style={styles.fileName}>{item}</Text>
          </Pressable>
        )}
      />
      <View style={styles.addFileContainer}>
        <TextInput
          style={styles.input}
          placeholder="new-file.js"
          value={newFileName}
          onChangeText={setNewFileName}
          onSubmitEditing={handleAddFile}
          autoCapitalize="none"
        />
        <Pressable style={styles.addButton} onPress={handleAddFile}>
          <Text style={styles.addButtonText}>Add File</Text>
        </Pressable>
      </View>
      <Pressable style={styles.templatesButton} onPress={() => router.push('/templates')}>
        <Text style={styles.templatesButtonText}>Browse Templates</Text>
      </Pressable>
      <Pressable style={styles.shareButton} onPress={() => router.push('/share')}>
        <Text style={styles.shareButtonText}>Share Project</Text>
      </Pressable>
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    paddingTop: 60,
    backgroundColor: '#252526',
  },
  header: {
    fontSize: 24,
    fontWeight: 'bold',
    color: '#fff',
    paddingHorizontal: 15,
    marginBottom: 20,
  },
  fileItem: {
    padding: 15,
    borderBottomWidth: 1,
    borderBottomColor: '#333',
  },
  activeFileItem: {
    backgroundColor: '#37373d',
  },
  fileName: {
    color: '#fff',
    fontSize: 16,
  },
  addFileContainer: {
    flexDirection: 'row',
    padding: 15,
    borderTopWidth: 1,
    borderTopColor: '#333',
  },
  input: {
    flex: 1,
    backgroundColor: '#3c3c3c',
    color: '#fff',
    padding: 10,
    borderRadius: 5,
    marginRight: 10,
  },
  addButton: {
    backgroundColor: '#6200ee',
    padding: 10,
    borderRadius: 5,
    justifyContent: 'center',
  },
  addButtonText: {
    color: '#fff',
    fontWeight: 'bold',
  },
  templatesButton: {
    backgroundColor: '#007aff',
    padding: 15,
    alignItems: 'center',
    justifyContent: 'center',
    margin: 15,
    borderRadius: 8,
  },
  templatesButtonText: {
    color: '#fff',
    fontSize: 16,
    fontWeight: 'bold',
  },
  shareButton: {
    backgroundColor: '#28a745',
    padding: 15,
    alignItems: 'center',
    justifyContent: 'center',
    marginHorizontal: 15,
    marginBottom: 15,
    borderRadius: 8,
  },
  shareButtonText: {
    color: '#fff',
    fontSize: 16,
    fontWeight: 'bold',
  },
});

export default FileTreeScreen;
