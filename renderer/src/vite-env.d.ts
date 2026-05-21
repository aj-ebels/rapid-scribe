/// <reference types="vite/client" />

interface RapidScribeApi {
  send: (type: string, params?: Record<string, unknown>) => Promise<unknown>;
  saveMarkdown: (defaultName: string, content: string) => Promise<{ saved: boolean; path?: string }>;
  openExternal: (url: string) => Promise<void>;
  on: (channel: string, callback: (data: unknown) => void) => () => void;
}

interface Window {
  api: RapidScribeApi;
}

interface Meeting {
  id: string;
  meeting_name: string;
  meeting_date?: string;
  manual_notes?: string;
  transcript?: string;
  ai_summary?: string;
  ai_chat_messages?: { role: string; content: string }[];
  created_at?: string;
  updated_at?: string;
}

interface Prompt {
  id: string;
  name: string;
  prompt: string;
}

interface Settings {
  audio_mode: string;
  meeting_mic_device: number | null;
  loopback_device_index: number | null;
  transcription_model?: string;
  auto_generate_summary_when_stopping?: boolean;
  export_prepend_meeting_date?: boolean;
  [key: string]: unknown;
}
