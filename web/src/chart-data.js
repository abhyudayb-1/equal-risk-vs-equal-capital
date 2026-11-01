export function windowRows(rows, window, decisions = false) {
  if (!window) return rows;
  const selected = rows.filter(
    (r) => r.date >= window.start && r.date <= window.end,
  );
  if (!decisions) return selected;
  // A monthly decision remains in force between closes; never import the next one.
  const anchor = rows.findLast((r) => r.date <= window.start);
  if (anchor && selected[0]?.date !== window.start)
    selected.unshift({ ...anchor, date: window.start });
  const last = selected.at(-1);
  if (last && last.date < window.end)
    selected.push({ ...last, date: window.end });
  return selected;
}
