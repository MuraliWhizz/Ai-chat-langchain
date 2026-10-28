import { existsSync } from "node:fs";
import { spawnSync } from "node:child_process";
import { fileURLToPath } from "node:url";
import path from "node:path";

const root = fileURLToPath(new URL("../", import.meta.url));
const isWindows = process.platform === "win32";
const npm = isWindows ? "npm.cmd" : "npm";
const python = isWindows ? "python" : "python3";
const pythonEnvironment = path.join(root, ".venv-docling");
const pythonExecutable = isWindows
  ? path.join(pythonEnvironment, "Scripts", "python.exe")
  : path.join(pythonEnvironment, "bin", "python");

function run(label, command, args, cwd = root) {
  console.log(`\n==> ${label}`);
  const result = spawnSync(command, args, {
    cwd,
    stdio: "inherit",
    shell: isWindows && command === npm,
  });

  if (result.error) {
    console.error(`Could not run ${label}: ${result.error.message}`);
    process.exit(result.status ?? 1);
  }

  if (result.status !== 0) {
    if (result.signal) {
      console.error(`${label} stopped by signal ${result.signal}.`);
    }
    process.exit(result.status ?? 1);
  }
}

if (process.argv[2] !== "install" || process.argv[3] !== "--all" || process.argv.length !== 4) {
  console.error("Usage: npm run cmd -- install --all");
  process.exit(1);
}

run("Installing root npm dependencies", npm, ["ci"]);
run("Installing API npm dependencies", npm, ["ci"], path.join(root, "apps", "api"));
run("Installing web npm dependencies", npm, ["ci"], path.join(root, "apps", "web"));

if (!existsSync(pythonExecutable)) {
  run("Creating the Docling Python environment", python, [
    "-m",
    "venv",
    pythonEnvironment,
  ]);
}

run("Installing Docling Python requirements", pythonExecutable, [
  "-m",
  "pip",
  "install",
  "-r",
  path.join(root, "services", "docling", "requirements.txt"),
]);

console.log("\nAll dependencies installed successfully.");
