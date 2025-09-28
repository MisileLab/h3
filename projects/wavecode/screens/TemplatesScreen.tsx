
import React from 'react';
import { View, Text, StyleSheet, FlatList, Pressable } from 'react-native';
import { templates } from '../constants/templates';
import { useProjectStore } from '../state/projectStore';
import { useRouter } from 'expo-router';

const TemplatesScreen = () => {
  const { activeFile, updateFile } = useProjectStore();
  const router = useRouter();

  const handleSelectTemplate = (templateContent: string) => {
    if (activeFile) {
      updateFile(activeFile, templateContent);
      router.back(); // Go back to the code screen after inserting
    } else {
      // Optionally, prompt the user to create a file first
      alert('Please select or create a file first.');
    }
  };

  return (
    <View style={styles.container}>
      <Text style={styles.title}>Templates & Snippets</Text>
      <FlatList
        data={templates}
        keyExtractor={(item) => item.name}
        renderItem={({ item }) => (
          <Pressable style={styles.templateItem} onPress={() => handleSelectTemplate(item.content)}>
            <Text style={styles.templateName}>{item.name}</Text>
            <Text style={styles.templateDescription}>{item.description}</Text>
          </Pressable>
        )}
      />
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
  templateItem: {
    backgroundColor: '#2d2d2d',
    padding: 15,
    borderRadius: 8,
    marginBottom: 10,
  },
  templateName: {
    fontSize: 18,
    fontWeight: 'bold',
    color: '#fff',
  },
  templateDescription: {
    fontSize: 14,
    color: '#ccc',
    marginTop: 5,
  },
  closeButton: {
    backgroundColor: '#007aff',
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

export default TemplatesScreen;
