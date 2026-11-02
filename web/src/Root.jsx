import React from "react";
import App from "./App.jsx";

// Was a two-route switch (the landing page and a private timeline workspace).
// The workspace was removed with the rest of the non-research features, so
// there is one page and no router dependency at all.
export default function Root() {
  return <App />;
}
