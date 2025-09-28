import { Stack } from 'expo-router';
import SettingsButton from '../components/SettingsButton';

export default function RootLayout() {
  return (
    <Stack>
      <Stack.Screen
        name="index"
        options={{
          headerTitle: 'Mobile Vibe Coder',
          headerRight: () => <SettingsButton />,
        }}
      />
      <Stack.Screen name="settings" options={{ presentation: 'modal', headerTitle: 'Settings' }} />
      <Stack.Screen name="templates" options={{ presentation: 'modal', headerTitle: 'Templates' }} />
      <Stack.Screen name="share" options={{ presentation: 'modal', headerTitle: 'Share Project' }} />
    </Stack>
  );
}
