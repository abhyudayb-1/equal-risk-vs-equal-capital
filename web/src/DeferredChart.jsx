import React, { lazy, Suspense, useEffect, useRef, useState } from "react";
const Chart = lazy(() => import("./Charts.jsx"));

class ChartBoundary extends React.Component {
  state = { failed: false };
  static getDerivedStateFromError() {
    return { failed: true };
  }
  render() {
    return this.state.failed ? (
      <p role="status">
        Chart could not load. Exported measurements remain visible.
      </p>
    ) : (
      this.props.children
    );
  }
}

export default function DeferredChart(props) {
  const ref = useRef();
  const [visible, setVisible] = useState(false);
  useEffect(() => {
    const observer = new IntersectionObserver(
      ([entry]) => {
        if (entry.isIntersecting) {
          setVisible(true);
          observer.disconnect();
        }
      },
      { rootMargin: "0px" },
    );
    observer.observe(ref.current);
    return () => observer.disconnect();
  }, []);
  const placeholder = (
    <div
      style={{ height: (props.height || 290) + 64 }}
      className="chart-skeleton"
    />
  );
  return (
    <div ref={ref} style={{ minHeight: (props.height || 290) + 64 }}>
      {visible && props.rows?.length ? (
        <ChartBoundary>
          <Suspense fallback={placeholder}>
            <Chart {...props} />
          </Suspense>
        </ChartBoundary>
      ) : (
        placeholder
      )}
    </div>
  );
}
