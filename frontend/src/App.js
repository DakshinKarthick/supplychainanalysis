import React, { useEffect, useState, useRef, useCallback } from 'react';
import 'leaflet/dist/leaflet.css';
import { MapContainer, TileLayer, Popup, Marker, Polyline, CircleMarker, useMap } from 'react-leaflet';
import L from 'leaflet';
import './App.css';

import iconImg from 'leaflet/dist/images/marker-icon.png';
import iconShadow from 'leaflet/dist/images/marker-shadow.png';

L.Marker.prototype.options.icon = L.icon({
  iconUrl: iconImg,
  shadowUrl: iconShadow,
  iconSize: [25, 41],
  iconAnchor: [12, 41],
});

const API = 'http://localhost:5050';

const ROUTE_COLORS = [
  '#e6194b', '#3cb44b', '#4363d8', '#f58231',
  '#911eb4', '#42d4f4', '#f032e6', '#bfef45',
];

/* ── Auto-fit map bounds ─────────────────────────────────────────────────── */
function FitBounds({ positions }) {
  const map = useMap();
  useEffect(() => {
    if (positions.length > 0) {
      map.fitBounds(positions, { padding: [30, 30] });
    }
  }, [map, positions]);
  return null;
}

/* ── Main App ────────────────────────────────────────────────────────────── */
export default function App() {
  const [lat, setLat] = useState('12.35');
  const [lon, setLon] = useState('78.55');
  const [milkQty, setMilkQty] = useState('0');
  const [runType, setRunType] = useState('insertion');
  const [logs, setLogs] = useState([]);
  const [status, setStatus] = useState('idle');
  const [phase, setPhase] = useState('');
  const [routes, setRoutes] = useState([]);
  const [cc, setCc] = useState(null);
  const [fullResults, setFullResults] = useState(null);
  const [insertionResults, setInsertionResults] = useState(null);
  const [newHmb, setNewHmb] = useState(null);
  const [selectedInsertionIdx, setSelectedInsertionIdx] = useState(0);
  const logRef = useRef(null);
  const eventSourceRef = useRef(null);

  /* scroll log to bottom */
  useEffect(() => {
    if (logRef.current) logRef.current.scrollTop = logRef.current.scrollHeight;
  }, [logs]);

  /* fetch initial routes on mount */
  useEffect(() => {
    fetch(`${API}/api/routes`)
      .then(r => r.json())
      .then(d => { setRoutes(d.routes || []); setCc(d.cc); })
      .catch(() => {});
  }, []);

  /* SSE listener */
  const connectSSE = useCallback(() => {
    if (eventSourceRef.current) eventSourceRef.current.close();

    const es = new EventSource(`${API}/api/events`);
    eventSourceRef.current = es;

    es.onmessage = (e) => {
      const msg = JSON.parse(e.data);

      switch (msg.type) {
        case 'log':
          setLogs(prev => [...prev, msg.message]);
          break;
        case 'status':
          setStatus(msg.status);
          break;
        case 'phase':
          setPhase(`${msg.phase} — ${msg.status}`);
          break;
        case 'insertion_results':
          setInsertionResults(msg.data);
          if (msg.new_hmb) setNewHmb(msg.new_hmb);
          break;
        case 'full_results':
          setFullResults(msg.data);
          if (msg.data?.new_hmb) setNewHmb(msg.data.new_hmb);
          if (msg.data?.cc) setCc(msg.data.cc);
          // Update map routes from full optimization
          if (msg.data?.routes) {
            setRoutes(msg.data.routes.map(r => ({
              route_code: r.route_code,
              route_name: r.route_name,
              hmbs: r.hmbs || [],
            })));
          }
          break;
        case 'error':
          setStatus('error');
          setLogs(prev => [...prev, `ERROR: ${msg.message}`]);
          break;
        default:
          break;
      }
    };

    es.onerror = () => { /* reconnects automatically */ };
    return es;
  }, []);

  /* kick off an optimization run */
  const handleRun = async () => {
    setLogs([]);
    setStatus('running');
    setPhase('');
    setInsertionResults(null);
    setFullResults(null);
    setNewHmb(null);
    setSelectedInsertionIdx(0);

    connectSSE();

    const body = { type: runType, mode: 'haversine' };

    if (lat.trim() && lon.trim()) {
      body.lat = parseFloat(lat);
      body.lon = parseFloat(lon);
    }
    if (milkQty.trim()) {
      body.milk_qty = parseFloat(milkQty);
    }

    await fetch(`${API}/api/run`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });
  };

  /* ── derive map data ───────────────────────────────────────────────────── */
  const displayRoutes = fullResults?.routes ?? routes;
  const allPositions = [];
  if (cc) allPositions.push([cc.lat, cc.lng]);
  displayRoutes.forEach(r => r.hmbs?.forEach(h => allPositions.push([h.lat, h.lng])));
  if (newHmb) allPositions.push([newHmb.lat, newHmb.lng]);

  const mapCenter = cc ? [cc.lat, cc.lng] : [12.3, 78.5];

  return (
    <div style={{ display: 'flex', height: '100vh', fontFamily: 'system-ui, sans-serif' }}>

      {/* ── Sidebar ─────────────────────────────────────────────────────── */}
      <div style={{
        width: 400, minWidth: 360, background: '#1e1e2e', color: '#cdd6f4',
        display: 'flex', flexDirection: 'column', overflow: 'hidden',
      }}>
        <div style={{ padding: '16px 16px 8px', borderBottom: '1px solid #313244' }}>
          <h3 style={{ margin: 0, color: '#89b4fa' }}>Route Optimizer</h3>
        </div>

        {/* controls */}
        <div style={{ padding: 16, borderBottom: '1px solid #313244' }}>
          <label style={{ fontSize: 13 }}>Optimization Type</label>
          <select value={runType} onChange={e => setRunType(e.target.value)}
            style={inputStyle}>
            <option value="insertion">Insert Route (2-Opt)</option>
            <option value="full">Full Route Optimization (GA)</option>
          </select>

          <label style={{ fontSize: 13 }}>
            New HMB Latitude {runType === 'full' && <span style={{color:'#6c7086'}}>(optional)</span>}
          </label>
          <input value={lat} onChange={e => setLat(e.target.value)}
            placeholder="e.g. 12.35" style={inputStyle} />

          <label style={{ fontSize: 13 }}>
            New HMB Longitude {runType === 'full' && <span style={{color:'#6c7086'}}>(optional)</span>}
          </label>
          <input value={lon} onChange={e => setLon(e.target.value)}
            placeholder="e.g. 78.55" style={inputStyle} />

          <label style={{ fontSize: 13 }}>Expected Milk (litres/day)</label>
          <input value={milkQty} onChange={e => setMilkQty(e.target.value)}
            placeholder="e.g. 100" style={inputStyle} />

          <button onClick={handleRun} disabled={status === 'running'}
            style={{
              width: '100%', padding: '10px', marginTop: 8,
              background: status === 'running' ? '#585b70' : '#89b4fa',
              color: '#1e1e2e', border: 'none', borderRadius: 6,
              fontWeight: 600, fontSize: 14, cursor: status === 'running' ? 'wait' : 'pointer',
            }}>
            {status === 'running' ? `Running… ${phase}` : 'Run Optimization'}
          </button>
        </div>

        {/* ── Results Panel ─────────────────────────────────────────────── */}
        <div style={{ flex: 1, overflow: 'auto' }}>

          {/* Full optimization results */}
          {fullResults && (
            <div style={{ padding: '12px 16px', borderBottom: '1px solid #313244' }}>
              <div style={{ color: '#a6e3a1', fontWeight: 600, marginBottom: 8, fontSize: 14 }}>
                Full Optimization Complete
              </div>
              <div style={statRow}>
                <span>Original</span><b>{fullResults.original_km} KM</b>
              </div>
              <div style={statRow}>
                <span>Optimized</span><b style={{color:'#a6e3a1'}}>{fullResults.optimized_km} KM</b>
              </div>
              <div style={statRow}>
                <span>Saved</span>
                <b style={{color:'#f9e2af'}}>{fullResults.saving_km} KM ({fullResults.saving_pct}%)</b>
              </div>

              {/* Per-route details */}
              <div style={{ marginTop: 12 }}>
                {fullResults.routes?.filter(r => r.hmb_count > 0).map((r, i) => (
                  <div key={r.route_code} style={{
                    padding: '8px', marginTop: 6, borderRadius: 6,
                    background: '#313244', fontSize: 12,
                    borderLeft: `3px solid ${ROUTE_COLORS[i % ROUTE_COLORS.length]}`,
                  }}>
                    <div style={{ fontWeight: 600, marginBottom: 4 }}>
                      {r.route_code} ({r.route_name}) — {r.hmb_count} HMBs
                    </div>
                    <div>{r.distance_km} KM ({r.diff_km >= 0 ? '+' : ''}{r.diff_km}) | {r.est_time_h}h
                      {!r.time_ok && <span style={{color:'#f38ba8'}}> ⚠ OVER TIME</span>}
                    </div>
                    <div style={{ color: '#a6adc8', marginTop: 4 }}>
                      CC → {r.sequence?.join(' → ')} → CC
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Insertion results */}
          {insertionResults && insertionResults.length > 0 && (
            <div style={{ padding: '12px 16px', borderBottom: '1px solid #313244' }}>
              <div style={{ color: '#fab387', fontWeight: 600, marginBottom: 8, fontSize: 14 }}>
                Insertion Results (Ranked)
              </div>

              {insertionResults.map((r, i) => (
                <div key={r.route_code}
                  onClick={() => setSelectedInsertionIdx(i)}
                  style={{
                    padding: '8px', marginTop: 6, borderRadius: 6,
                    background: i === 0 ? '#2a3a2a' : '#313244', fontSize: 12,
                    borderLeft: `3px solid ${i === 0 ? '#a6e3a1' : '#585b70'}`,
                    cursor: 'pointer',
                  }}>
                  <div style={{ fontWeight: 600, marginBottom: 2 }}>
                    {i === 0 ? '#1 ' : `#${i+1} `}
                    {r.route_code} ({r.route_name})
                    {!r.feasible && <span style={{color:'#f38ba8'}}> {r.reason}</span>}
                  </div>
                  <div>
                    +{r.extra_km} KM | Post-2opt: {r.post_2opt_km} KM | Time: {r.est_time_h}h | Score: {r.score}
                  </div>
                  <div style={{ color: '#a6adc8', marginTop: 4 }}>
                    Insert: {r.prev_stop} → <b style={{color:'#f9e2af'}}>NEW</b> → {r.next_stop}
                  </div>
                  {r.sequence && (
                    <div style={{ color: '#a6adc8', marginTop: 2 }}>
                      CC → {r.sequence.join(' → ')} → CC
                    </div>
                  )}
                </div>
              ))}
            </div>
          )}

          {/* live log */}
          <div ref={logRef} style={{
            padding: '8px 12px',
            fontFamily: '"SF Mono", "Fira Code", monospace', fontSize: 11,
            lineHeight: 1.6, background: '#181825', minHeight: 100,
          }}>
            {logs.length === 0 && <span style={{ color: '#6c7086' }}>Logs will appear here…</span>}
            {logs.map((l, i) => (
              <div key={i} style={{ color: l.includes('ERROR') ? '#f38ba8' : '#a6adc8' }}>{l}</div>
            ))}
          </div>
        </div>
      </div>

      {/* ── Map ─────────────────────────────────────────────────────────── */}
      <div style={{ flex: 1 }}>
        <MapContainer center={mapCenter} zoom={10} scrollWheelZoom style={{ height: '100%', width: '100%' }}>
          <TileLayer
            attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a>'
            url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
          />
          {allPositions.length > 1 && <FitBounds positions={allPositions} />}

          {/* CC marker */}
          {cc && (
            <Marker position={[cc.lat, cc.lng]}>
              <Popup><b>CC (Uthangarai Chilling Center)</b></Popup>
            </Marker>
          )}

          {/* New HMB marker */}
          {newHmb && (
            <CircleMarker center={[newHmb.lat, newHmb.lng]} radius={10}
              pathOptions={{ color: '#f38ba8', fillColor: '#f38ba8', fillOpacity: 0.9 }}>
              <Popup><b>NEW HMB</b></Popup>
            </CircleMarker>
          )}

          {/* Route polylines + HMB markers */}
          {displayRoutes.map((route, idx) => {
            if (!route.hmbs || route.hmbs.length === 0) return null;
            const color = ROUTE_COLORS[idx % ROUTE_COLORS.length];

            // Build polyline points — if this route is the selected insertion route,
            // insert the new HMB at the correct sequence position
            const selResult = insertionResults && insertionResults[selectedInsertionIdx];
            const isSelectedRoute = selResult && route.route_code === selResult.route_code;
            const points = [];
            if (cc) points.push([cc.lat, cc.lng]);
            if (isSelectedRoute && selResult?.sequence && newHmb) {
              const hmbByName = {};
              route.hmbs.forEach(h => { hmbByName[h.name] = h; });
              selResult.sequence.forEach(name => {
                if (name === 'NEW HMB') {
                  points.push([newHmb.lat, newHmb.lng]);
                } else if (hmbByName[name]) {
                  points.push([hmbByName[name].lat, hmbByName[name].lng]);
                }
              });
            } else {
              route.hmbs.forEach(h => points.push([h.lat, h.lng]));
            }
            if (cc) points.push([cc.lat, cc.lng]);

            return (
              <React.Fragment key={route.route_code || idx}>
                <Polyline positions={points}
                  pathOptions={{ color, weight: 3, opacity: 0.8 }} />
                {route.hmbs.map((h, hi) => (
                  <CircleMarker key={hi} center={[h.lat, h.lng]} radius={5}
                    pathOptions={{ color, fillColor: color, fillOpacity: 0.8 }}>
                    <Popup>
                      <b>{h.name}</b><br />
                      Route: {route.route_code} ({route.route_name})
                      {h.from_route && <><br /><i>Moved from {h.from_route}</i></>}
                    </Popup>
                  </CircleMarker>
                ))}
              </React.Fragment>
            );
          })}
        </MapContainer>
      </div>
    </div>
  );
}

const inputStyle = {
  width: '100%', padding: '8px', marginBottom: 8, marginTop: 2,
  background: '#313244', color: '#cdd6f4', border: '1px solid #45475a',
  borderRadius: 4, fontSize: 13, boxSizing: 'border-box',
};

const statRow = {
  display: 'flex', justifyContent: 'space-between', fontSize: 13, marginBottom: 2,
};