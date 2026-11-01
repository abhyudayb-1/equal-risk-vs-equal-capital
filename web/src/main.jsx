import React from "react";
import { createRoot, hydrateRoot } from "react-dom/client";
import "@fontsource/inter/latin-400.css";
import "@fontsource/inter/latin-500.css";
import "@fontsource/inter/latin-600.css";
import "@fontsource/ibm-plex-mono/latin-400.css";
import "./theme.css";
import "./style.css";
import Root from "./Root.jsx";

const tree = (
  <React.StrictMode>
    <Root />
  </React.StrictMode>
);
// dist/index.html is prerendered from this one page (see prerender.mjs), so
// hydrating against it is always the right move when the snapshot is present.
// The path check this used to carry existed only because a second route (the
// removed timeline workspace) was served the same SPA fallback and would have
// hydrated against markup for a different page.
if (globalThis.__PROJECT_SNAPSHOT__)
  hydrateRoot(document.getElementById("root"), tree, {
    onRecoverableError: (error, info) =>
      console.error(error.message, info.componentStack),
  });
else createRoot(document.getElementById("root")).render(tree);
