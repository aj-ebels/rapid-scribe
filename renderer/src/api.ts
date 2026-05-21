export function isElectron(): boolean {
  return typeof window !== "undefined" && !!window.api;
}

function requireApi(): RapidScribeApi {
  if (!window.api) {
    throw new Error(
      "Rapid Scribe must run in the Electron desktop window (npm run dev), not in a regular browser tab at localhost:5173."
    );
  }
  return window.api;
}

export const api = new Proxy({} as RapidScribeApi, {
  get(_target, prop: keyof RapidScribeApi) {
    const a = requireApi();
    const value = a[prop];
    if (typeof value === "function") {
      return (value as (...args: unknown[]) => unknown).bind(a);
    }
    return value;
  },
});

export async function cmd<T = unknown>(
  type: string,
  params?: Record<string, unknown>
): Promise<T> {
  return (await requireApi().send(type, params)) as T;
}
