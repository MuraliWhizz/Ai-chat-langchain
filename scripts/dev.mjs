import { spawn } from "node:child_process";

const isWindows = process.platform === "win32";
const npm = isWindows ? "npm.cmd" : "npm";
const apiPort = process.env.API_PORT ?? "3001";
const webPort = process.env.WEB_PORT ?? "3000";

for (const [name, port] of [
  ["API_PORT", apiPort],
  ["WEB_PORT", webPort],
]) {
  if (!/^\d+$/.test(port) || Number(port) < 1 || Number(port) > 65535) {
    console.error(`${name} must be a valid TCP port (1-65535); received "${port}".`);
    process.exit(1);
  }
}

console.log(`Starting NestJS API on http://localhost:${apiPort}`);
console.log(`Starting Next.js web app on http://localhost:${webPort}`);
console.log("Press Ctrl+C to stop both apps.");

const children = [
  spawn(npm, ["run", "start:dev"], {
    cwd: new URL("../apps/api/", import.meta.url),
    env: { ...process.env, PORT: apiPort },
    stdio: "inherit",
    detached: !isWindows,
    shell: isWindows,
  }),
  spawn(npm, ["run", "dev", "--", "--port", webPort], {
    cwd: new URL("../apps/web/", import.meta.url),
    env: process.env,
    stdio: "inherit",
    detached: !isWindows,
    shell: isWindows,
  }),
];

let stopping = false;

function stopChild(child, signal) {
  if (child.exitCode !== null || child.signalCode !== null || !child.pid) return;

  try {
    if (isWindows) {
      child.kill(signal);
    } else {
      process.kill(-child.pid, signal);
    }
  } catch (error) {
    if (error.code !== "ESRCH") throw error;
  }
}

function stopAll(signal = "SIGTERM", exitCode = 0) {
  if (stopping) return;
  stopping = true;
  for (const child of children) stopChild(child, signal);

  const forceStop = setTimeout(() => {
    for (const child of children) stopChild(child, "SIGKILL");
  }, 5000);
  forceStop.unref();

  Promise.all(
    children.map(
      (child) =>
        new Promise((resolve) => {
          if (child.exitCode !== null || child.signalCode !== null) {
            resolve();
          } else {
            child.once("exit", resolve);
          }
        }),
    ),
  ).finally(() => {
    clearTimeout(forceStop);
    process.exit(exitCode);
  });
}

for (const child of children) {
  child.once("error", (error) => {
    console.error(`Could not start an application process: ${error.message}`);
    stopAll("SIGTERM", 1);
  });
  child.once("exit", (code, signal) => {
    if (stopping) return;
    if (code !== 0) {
      console.error(
        `An application stopped unexpectedly (${signal ?? `exit code ${code ?? "unknown"}`}); stopping both.`,
      );
      stopAll("SIGTERM", code ?? 1);
    } else {
      stopAll("SIGTERM", 0);
    }
  });
}

process.once("SIGINT", () => stopAll("SIGINT", 130));
process.once("SIGTERM", () => stopAll("SIGTERM", 143));
