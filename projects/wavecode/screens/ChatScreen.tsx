import React, { useState, useCallback } from 'react';
import { GiftedChat, IMessage } from 'react-native-gifted-chat';
import { getCodeFromAI } from '../services/OpenRouter';
import { useProjectStore } from '../state/projectStore';
import { SafeAreaView } from 'react-native-safe-area-context';
import { OPENROUTER_API_KEY } from '@env';

const ChatScreen = () => {
  const [messages, setMessages] = useState<IMessage[]>([]);
  const { files, activeFile, updateFile, model } = useProjectStore();

  React.useEffect(() => {
    const missingApiKeyMessage: IMessage = {
      _id: 0,
      text: `Welcome to Mobile Vibe Coder!\n\n**Warning:** No API key found.\n\nThe AI is currently in mock mode. To enable real AI responses, please add your OpenRouter key to the .env file and restart the application.`,      createdAt: new Date(),
      user: { _id: 2, name: 'System' },
    };

    const initialMessages = (!OPENROUTER_API_KEY || OPENROUTER_API_KEY === 'your_openrouter_api_key_here')
      ? [missingApiKeyMessage]
      : [{
          _id: 1,
          text: 'Hello! I am your AI coding assistant. How can I help you today?',
          createdAt: new Date(),
          user: { _id: 2, name: 'AI Assistant' },
        }];
    setMessages(initialMessages);
  }, []);

  const onSend = useCallback(async (newMessages: IMessage[] = []) => {
    const userMessage = newMessages[0];
    setMessages(previousMessages =>
      GiftedChat.append(previousMessages, userMessage),
    );

    if (!activeFile) {
      const errorMessage: IMessage = {
        _id: new Date().getTime(),
        text: 'Please select a file to edit first.',
        createdAt: new Date(),
        user: { _id: 2, name: 'AI Assistant' },
      };
      setMessages(previousMessages =>
        GiftedChat.append(previousMessages, errorMessage),
      );
      return;
    }

    const prompt = userMessage.text;
    const context = files[activeFile];

    // 1. Create a placeholder message for the streaming response
    const streamingMessageId = new Date().getTime();
    const placeholderMessage: IMessage = {
      _id: streamingMessageId,
      text: '…',
      createdAt: new Date(),
      user: { _id: 2, name: 'AI Assistant' },
    };
    setMessages(previousMessages =>
      GiftedChat.append(previousMessages, placeholderMessage),
    );

    let streamedResponse = '';

    getCodeFromAI({
      prompt, context, fileName: activeFile, model,
      onChunk: (chunk) => {
        streamedResponse += chunk;
        // 2. Update the message content as chunks arrive
        setMessages(previousMessages => {
          const msgIndex = previousMessages.findIndex(msg => msg._id === streamingMessageId);
          if (msgIndex === -1) return previousMessages;
          const updatedMessages = [...previousMessages];
          updatedMessages[msgIndex] = {
            ...updatedMessages[msgIndex],
            text: streamedResponse,
          };
          return updatedMessages;
        });
      },
      onComplete: (fullResponse) => {
        // 3. Extract the final code and update the file
        const codeBlockMatch = fullResponse.match(/```(?:[a-z]+)?\n([\s\S]*?)```/);
        const finalCode = codeBlockMatch ? codeBlockMatch[1].trim() : '';

        if (finalCode && finalCode !== context.trim()) {
          updateFile(activeFile, finalCode);
        } else if (!finalCode) {
            console.log("AI did not return a code block.");
        }
      },
      onError: (error) => {
        setMessages(previousMessages => {
          const msgIndex = previousMessages.findIndex(msg => msg._id === streamingMessageId);
          if (msgIndex === -1) return previousMessages;
          const updatedMessages = [...previousMessages];
          updatedMessages[msgIndex] = {
            ...updatedMessages[msgIndex],
            text: `Sorry, an error occurred: ${error.message}`,
          };
          return updatedMessages;
        });
      },
    });

  }, [activeFile, files, updateFile, model]);

  return (
    <SafeAreaView style={{ flex: 1 }}>
      <GiftedChat
        messages={messages}
        onSend={messages => onSend(messages)}
        user={{
          _id: 1,
        }}
        messagesContainerStyle={{ paddingTop: 40 }}
      />
    </SafeAreaView>
  );
};

export default ChatScreen;