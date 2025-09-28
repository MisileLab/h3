
import { View, Text, StyleSheet, Pressable, TextInput } from 'react-native';
import { useProjectStore, availableModels } from '../state/projectStore';
import { useRouter } from 'expo-router';

const SettingsScreen = () => {
  const { model, setModel } = useProjectStore();
  const router = useRouter();

  return (
    <View style={styles.container}>
      <Text style={styles.title}>Settings</Text>

      <Text style={styles.label}>AI Model</Text>
      <TextInput
        style={styles.input}
        value={model}
        onChangeText={setModel}
        placeholder="Enter custom model name"
        autoCapitalize="none"
      />

      <Text style={styles.label}>Presets</Text>
      <View style={styles.pickerContainer}>
        {availableModels.map((m) => (
          <Pressable
            key={m}
            style={[styles.pickerItem, model === m && styles.activeItem]}
            onPress={() => setModel(m)}>
            <Text style={styles.pickerText}>{m}</Text>
          </Pressable>
        ))}
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
  label: {
    fontSize: 18,
    color: '#fff',
    marginBottom: 10,
    marginTop: 10,
  },
  input: {
    backgroundColor: '#2d2d2d',
    color: '#fff',
    padding: 15,
    borderRadius: 8,
    fontSize: 16,
  },
  pickerContainer: {
    backgroundColor: '#2d2d2d',
    borderRadius: 8,
  },
  pickerItem: {
    padding: 15,
    borderBottomWidth: 1,
    borderBottomColor: '#424242',
  },
  activeItem: {
    backgroundColor: '#007aff',
  },
  pickerText: {
    fontSize: 16,
    color: '#fff',
  },
  closeButton: {
    backgroundColor: '#007aff',
    padding: 15,
    borderRadius: 8,
    alignItems: 'center',
    marginTop: 'auto',
  },
  closeButtonText: {
    color: '#fff',
    fontSize: 16,
    fontWeight: 'bold'
  },
});

export default SettingsScreen;
