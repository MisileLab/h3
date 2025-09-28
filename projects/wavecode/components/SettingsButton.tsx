
import React from 'react';
import { Pressable } from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import { useRouter } from 'expo-router';

const SettingsButton = () => {
  const router = useRouter();
  return (
    <Pressable onPress={() => router.push('/settings')}>
      <Ionicons name="settings" size={24} color="black" />
    </Pressable>
  );
};

export default SettingsButton;
