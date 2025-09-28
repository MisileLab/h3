import { OPENROUTER_API_KEY } from '@env';

interface GetCodeFromAIParams {
  prompt: string;
  fileContent: string;
  fileName: string;
  model: string;
  onChunk: (chunk: string) => void;
  onComplete: (fullResponse: string) => void;
  onError: (error: Error) => void;
}

export const getCodeFromAI = async ({ prompt, fileContent, fileName, model, onChunk, onComplete, onError }: GetCodeFromAIParams): Promise<void> => {
  const systemPrompt = `You are an expert code editor. The user is currently editing a file named "${fileName}".

The current content of the file is:
\`\`\`
${fileContent}
\`\`\`

The user wants to make the following change: "${prompt}".

Your task is to first, think step-by-step and explain your plan. Then, after you have explained, output the *entire*, updated code for the file in a single markdown code block. Do not add any explanations after the code block.`;

  if (!OPENROUTER_API_KEY || OPENROUTER_API_KEY === 'your_openrouter_api_key_here') {
    const error = new Error('Please set your OPENROUTER_API_KEY in the .env file.');
    onError(error);
    return;
  }

  try {
    const response = await fetch("https://openrouter.ai/api/v1/chat/completions", {
      method: 'POST',
      headers: {
        'Authorization': `Bearer ${OPENROUTER_API_KEY}`,
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        model: model,
        messages: [
          { role: 'system', content: systemPrompt },
          { role: 'user', content: prompt },
        ],
        stream: false, // Disabled streaming due to React Native limitations
      }),
    });

    if (!response.ok) {
      const errorBody = await response.text();
      console.error("OpenRouter Error Response:", errorBody);
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    const data = await response.json();
    if (data.choices && data.choices[0] && data.choices[0].message && data.choices[0].message.content) {
        const fullResponse = data.choices[0].message.content;
        onChunk(fullResponse);
        onComplete(fullResponse);
    } else {
        console.error('Unexpected API response format:', data);
        throw new Error('Unexpected API response format');
    }

  } catch (error) {
    console.error('Error calling OpenRouter API:', error);
    onError(error as Error);
  }
};
