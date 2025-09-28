import { create } from 'zustand';
import { createJSONStorage, persist } from 'zustand/middleware';
import AsyncStorage from '@react-native-async-storage/async-storage';

export interface ProjectFile {
  name: string;
  content: string;
}

export const availableModels = [
  'deepseek/deepseek-chat-v3.1:free',
  'google/gemini-flash-1.5',
  'openai/gpt-4o',
  'anthropic/claude-3.5-sonnet',
  'meta-llama/llama-3-70b-instruct',
];

interface ProjectState {
  files: Record<string, string>;
  activeFile: string | null;
  model: string;
  setModel: (model: string) => void;
  addFile: (filename: string) => void;
  updateFile: (filename: string, content: string) => void;
  setActiveFile: (filename: string | null) => void;
  exportProject: () => string;
  importProject: (jsonString: string) => void;
}

const initialCode = `<h1>Welcome to Mobile Vibe Coder!</h1>
<p>Ask the AI to build something. You can also create new files by swiping right to the file explorer.</p>
<link rel="stylesheet" href="style.css">`;
const initialCss = `body {
  font-family: sans-serif;
  background-color: #f0f0f0;
  color: #333;
  display: flex;
  justify-content: center;
  align-items: center;
  height: 100vh;
  text-align: center;
}`;

export const useProjectStore = create<ProjectState>()(
  persist(
    (set, get) => ({
      files: {
        'index.html': initialCode,
        'style.css': initialCss,
      },
      activeFile: 'index.html',
      model: availableModels[0],
      setModel: (model) => set({ model }),
      addFile: (filename) => {
        if (get().files[filename]) {
          console.warn(`File "${filename}" already exists.`);
          return;
        }
        set((state) => ({
          files: { ...state.files, [filename]: '// New file' },
          activeFile: filename,
        }));
      },
      updateFile: (filename, content) => {
        set((state) => ({
          files: { ...state.files, [filename]: content },
        }));
      },
      setActiveFile: (filename) => {
        if (filename && !get().files[filename]) {
          console.warn(`File "${filename}" does not exist.`);
          return;
        }
        set({ activeFile: filename });
      },
      exportProject: () => {
        const state = get();
        return JSON.stringify({
          files: state.files,
          activeFile: state.activeFile,
          model: state.model,
        });
      },
      importProject: (jsonString: string) => {
        try {
          const importedState = JSON.parse(jsonString);
          set({
            files: importedState.files || {},
            activeFile: importedState.activeFile || null,
            model: importedState.model || availableModels[0],
          });
        } catch (error) {
          console.error('Error importing project:', error);
          alert('Invalid project data.');
        }
      },
    }),
    {
      name: 'mobile-vibe-coder-storage-multifile',
      storage: createJSONStorage(() => AsyncStorage),
    }
  )
);
