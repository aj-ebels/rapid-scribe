import { EventEmitter } from "events";
import { ChildProcess, spawn } from "child_process";
import * as fs from "fs";
import * as path from "path";

export interface SidecarMessage {
  id?: string;
  type: string;
  ok?: boolean;
  result?: unknown;
  error?: string;
  data?: Record<string, unknown>;
}

export class SidecarClient extends EventEmitter {
  private proc: ChildProcess | null = null;
  private buffer = "";
  private pending = new Map<
    string,
    { resolve: (v: unknown) => void; reject: (e: Error) => void }
  >();
  private nextId = 1;

  constructor(
    private command: string,
    private args: string[] = []
  ) {
    super();
  }

  start(): Promise<void> {
    return new Promise((resolve, reject) => {
      const isPython = this.command.toLowerCase().includes("python");
      if (!isPython && !fs.existsSync(this.command)) {
        reject(new Error(`Sidecar not found: ${this.command}`));
        return;
      }
      let settled = false;
      const env = {
        ...process.env,
        ...(process.env.NODE_ENV === "development" ? { MEETINGS_SIDECAR_LOG: "1" } : {}),
      };
      this.proc = spawn(this.command, this.args, {
        stdio: ["pipe", "pipe", "pipe"],
        windowsHide: true,
        cwd: isPython ? process.cwd() : path.dirname(this.command),
        env,
      });
      const onReady = (msg: SidecarMessage) => {
        if (msg.type === "ready") {
          settled = true;
          this.off("message", onReady);
          resolve();
        }
      };
      this.on("message", onReady);
      this.proc.stdout?.on("data", (chunk: Buffer) => this._onData(chunk.toString()));
      this.proc.stderr?.on("data", (chunk: Buffer) => {
        this.emit("stderr", chunk.toString());
      });
      this.proc.on("exit", (code) => {
        this.emit("exit", code);
        for (const [, p] of this.pending) {
          p.reject(new Error(`Sidecar exited (${code})`));
        }
        this.pending.clear();
      });
      this.proc.on("error", reject);
      setTimeout(() => {
        if (!settled) reject(new Error("Sidecar startup timeout"));
      }, 120000);
    });
  }

  _onData(chunk: string) {
    this.buffer += chunk;
    let idx: number;
    while ((idx = this.buffer.indexOf("\n")) >= 0) {
      const line = this.buffer.slice(0, idx).trim();
      this.buffer = this.buffer.slice(idx + 1);
      if (!line) continue;
      try {
        const msg = JSON.parse(line) as SidecarMessage;
        this._dispatch(msg);
      } catch {
        /* ignore malformed */
      }
    }
  }

  private _dispatch(msg: SidecarMessage) {
    this.emit("message", msg);
    if (msg.type === "response" && msg.id) {
      const p = this.pending.get(msg.id);
      if (p) {
        this.pending.delete(msg.id);
        if (msg.ok) p.resolve(msg.result);
        else p.reject(new Error(msg.error || "Sidecar error"));
      }
      return;
    }
    if (!["response", "ready"].includes(msg.type)) {
      this.emit(`event:${msg.type}`, msg.data ?? {}, msg.id);
    }
  }

  send(type: string, params: Record<string, unknown> = {}): Promise<unknown> {
    const id = `r${this.nextId++}`;
    return new Promise((resolve, reject) => {
      if (!this.proc?.stdin?.writable) {
        reject(new Error("Sidecar not running"));
        return;
      }
      this.pending.set(id, { resolve, reject });
      const line = JSON.stringify({ id, type, params }) + "\n";
      this.proc.stdin.write(line, (err) => {
        if (err) {
          this.pending.delete(id);
          reject(err);
        }
      });
      setTimeout(() => {
        if (this.pending.has(id)) {
          this.pending.delete(id);
          reject(new Error(`Timeout: ${type}`));
        }
      }, 300000);
    });
  }

  stop(): void {
    if (this.proc) {
      try {
        this.send("shutdown", {}).catch(() => undefined);
      } catch {
        /* */
      }
      this.proc.kill();
      this.proc = null;
    }
  }
}

export function resolveSidecarLaunch(): { command: string; args: string[] } {
  const isDev = process.env.NODE_ENV === "development";
  if (isDev) {
    const devExe = path.join(
      process.cwd(),
      "dist",
      "Rapid Scribe Sidecar",
      "rapid-scribe-sidecar.exe"
    );
    if (fs.existsSync(devExe)) {
      return { command: devExe, args: [] };
    }
    const venvPython = path.join(process.cwd(), ".venv", "Scripts", "python.exe");
    if (fs.existsSync(venvPython)) {
      return { command: venvPython, args: ["-m", "sidecar.sidecar"] };
    }
    return { command: "python", args: ["-m", "sidecar.sidecar"] };
  }
  const exe = path.join(process.resourcesPath, "sidecar", "rapid-scribe-sidecar.exe");
  return { command: exe, args: [] };
}
