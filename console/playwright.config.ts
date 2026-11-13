import { defineConfig } from "@playwright/test";
export default defineConfig({
  testDir: "./tests",
  timeout: 90000,
  expect: { timeout: 15000 },
  workers: 1,
  retries: 0,
  reporter: [["list"], ["html", { open: "never" }]],
  use: {
    baseURL: "http://127.0.0.1:3102",
    trace: "retain-on-failure",
    screenshot: "only-on-failure",
    launchOptions: process.env.WORKBENCH_CHROME
      ? { executablePath: process.env.WORKBENCH_CHROME }
      : {},
  },
  webServer: [
    {
      command: `${process.env.WORKBENCH_TEST_PYTHON || "python3"} ../tools/console_fixture.py`,
      url: "http://127.0.0.1:8768/api/health",
      timeout: 90000,
      reuseExistingServer: false,
    },
    {
      command:
        "node node_modules/next/dist/bin/next start --hostname 127.0.0.1 --port 3102",
      url: "http://127.0.0.1:3102",
      timeout: 90000,
      reuseExistingServer: false,
      env: {
        WORKBENCH_API_URL: "http://127.0.0.1:8768",
        WORKBENCH_CONSOLE_TOKEN: "fixture-operator-token-not-for-real-use",
      },
    },
  ],
});
