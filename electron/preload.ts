import { contextBridge, ipcRenderer } from "electron";

const api = {
  send: (type: string, params?: Record<string, unknown>) =>
    ipcRenderer.invoke("sidecar:send", type, params ?? {}),
  saveMarkdown: (defaultName: string, content: string) =>
    ipcRenderer.invoke("dialog:saveMarkdown", defaultName, content),
  openExternal: (url: string) => ipcRenderer.invoke("shell:openExternal", url),
  on: (channel: string, callback: (data: unknown) => void) => {
    const sub = (_: unknown, data: unknown) => callback(data);
    ipcRenderer.on(channel, sub);
    return () => ipcRenderer.removeListener(channel, sub);
  },
};

contextBridge.exposeInMainWorld("api", api);

export type RapidScribeApi = typeof api;
