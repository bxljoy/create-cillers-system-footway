import { ApiClientInterface } from '../types';

export interface Response {
  message: string;
}

export interface Message {
  role: 'user' | 'bot';
  content: string;
}

export function createChatApi(client: ApiClientInterface) {
  return {
    async sendMessage(prompt: string): Promise<Message> {
      debugger;
      const on_error = (messages: string[]) => {
        console.error('Error calling the API:', messages);
      };
      var response = await client.post<Response>('/api/test/anthropic', on_error, { prompt });
      return { role: 'bot', content: response.message };
    },
  };
}
