import {
  app,
  BrowserWindow,
  ipcMain,
  dialog,
  shell,
} from "electron";
import * as path from "path";
import * as fs from "fs";
import { autoUpdater } from "electron-updater";
import { SidecarClient, resolveSidecarLaunch } from "./sidecar-client";

let mainWindow: BrowserWindow | null = null;
let sidecar: SidecarClient | null = null;

const isDev = process.env.NODE_ENV === "development";

if (!isDev) {
  const gotLock = app.requestSingleInstanceLock();
  if (!gotLock) {
    app.quit();
  } else {
    app.on("second-instance", () => {
      if (mainWindow) {
        if (mainWindow.isMinimized()) mainWindow.restore();
        mainWindow.focus();
      }
    });
  }
}

function getSidecarLogTail(maxLines = 40): string {
  const appData = process.env.APPDATA || "";
  const logPath = path.join(appData, "Meetings", "sidecar.log");
  try {
    const text = fs.readFileSync(logPath, "utf8");
    return text.split("\n").slice(-maxLines).join("\n");
  } catch {
    return "";
  }
}

function forwardSidecarEvents(client: SidecarClient) {
  const events = [
    "transcript_line",
    "audio_level",
    "status",
    "capture_error",
    "model_progress",
    "summary_chunk",
    "summary_done",
    "ask_chunk",
    "ask_done",
  ];
  for (const ev of events) {
    client.on(`event:${ev}`, (data: unknown) => {
      mainWindow?.webContents.send(`sidecar:${ev}`, data);
    });
  }
  client.on("exit", (code: number) => {
    mainWindow?.webContents.send("sidecar:exit", { code, log: getSidecarLogTail() });
  });
}

function createWindow() {
  mainWindow = new BrowserWindow({
    width: 1280,
    height: 860,
    minWidth: 900,
    minHeight: 600,
    show: false,
    title: "Rapid Scribe",
    backgroundColor: "#1e1e2e",
    webPreferences: {
      preload: path.join(__dirname, "preload.js"),
      contextIsolation: true,
      nodeIntegration: false,
    },
  });

  mainWindow.once("ready-to-show", () => {
    mainWindow?.show();
    mainWindow?.focus();
  });

  if (isDev) {
    mainWindow.loadURL("http://127.0.0.1:5173");
    // DevTools autofill warnings are harmless; open manually with F12 if needed.
  } else {
    mainWindow.loadFile(path.join(__dirname, "..", "dist-renderer", "index.html"));
  }

  mainWindow.on("closed", () => {
    mainWindow = null;
  });
}

app.whenReady().then(async () => {
  createWindow();

  const launch = resolveSidecarLaunch();
  sidecar = new SidecarClient(launch.command, launch.args);
  try {
    await sidecar.start();
    await sidecar.send("ping", {});
    forwardSidecarEvents(sidecar);
    mainWindow?.webContents.send("sidecar:ready", { ok: true });
  } catch (e) {
    const err = e instanceof Error ? e.message : String(e);
    mainWindow?.webContents.send("sidecar:ready", {
      ok: false,
      error: err,
      log: getSidecarLogTail(),
    });
    dialog.showErrorBox(
      "Rapid Scribe",
      `Failed to start the audio engine.\n\n${err}\n\n${getSidecarLogTail()}`
    );
  }

  if (app.isPackaged) {
    autoUpdater.checkForUpdatesAndNotify().catch(() => undefined);
  }
});

app.on("window-all-closed", () => {
  sidecar?.stop();
  if (process.platform !== "darwin") app.quit();
});

app.on("before-quit", () => {
  sidecar?.stop();
});

ipcMain.handle("sidecar:send", async (_ev, type: string, params: Record<string, unknown>) => {
  if (!sidecar) throw new Error("Sidecar not running");
  return sidecar.send(type, params);
});

ipcMain.handle("dialog:saveMarkdown", async (_ev, defaultName: string, content: string) => {
  const win = BrowserWindow.getFocusedWindow() || mainWindow;
  const { canceled, filePath } = await dialog.showSaveDialog(win!, {
    defaultPath: defaultName.endsWith(".md") ? defaultName : `${defaultName}.md`,
    filters: [{ name: "Markdown", extensions: ["md"] }],
  });
  if (canceled || !filePath) return { saved: false };
  fs.writeFileSync(filePath, content, "utf8");
  return { saved: true, path: filePath };
});

ipcMain.handle("shell:openExternal", async (_ev, url: string) => {
  await shell.openExternal(url);
});
