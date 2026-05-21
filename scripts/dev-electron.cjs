// Wait for Vite, then launch Electron (reliable on Windows; avoids shell && chains).
const { spawn } = require("child_process");
const waitOn = require("wait-on");
const electron = require("electron");

waitOn({
  resources: ["tcp:127.0.0.1:5173"],
  timeout: 120000,
  interval: 250,
  window: 1000,
})
  .then(() => {
    console.log("[electron] Vite is up — starting desktop window…");
    const env = { ...process.env, NODE_ENV: "development" };
    const child = spawn(electron, ["."], {
      stdio: "inherit",
      env,
      cwd: process.cwd(),
    });
    child.on("exit", (code) => {
      console.log("[electron] Desktop app closed.");
      process.exit(code ?? 0);
    });
  })
  .catch((err) => {
    console.error("[electron] Timed out waiting for Vite at 127.0.0.1:5173:", err.message);
    process.exit(1);
  });
