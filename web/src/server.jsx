import React from "react";
import { renderToString } from "react-dom/server";
import App from "./App.jsx";

export function render(snapshot) {
  globalThis.__PROJECT_SNAPSHOT__ = snapshot;
  try {
    return renderToString(<App />);
  } finally {
    delete globalThis.__PROJECT_SNAPSHOT__;
  }
}
