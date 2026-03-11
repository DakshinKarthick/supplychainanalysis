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
  const [runType, setRunType] = useState('both');
  const [logs, setLogs] = useState([]);
  const [status, setStatus] = useState('idle');        // idle | running | complete | error
  const [phase, setPhase] = useState('');
  const [routes, setRoutes] = useState([]);             // initial routes
  const [cc, setCc] = useState(null);                   // CC coords
  const [fullResults, setFullResults] = useState(null);  // optimized routes
  const [insertionResults, setInsertionResults] = useState(null);
  const [newHmb, setNewHmb] = useState(null);
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

  /* SSE listener — connect once and keep open */
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

    connectSSE();

    await fetch(`${API}/api/run`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ lat: parseFloat(lat), lon: parseFloat(lon), type: runType }),
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
        width: 380, minWidth: 340, background: '#1e1e2e', color: '#cdd6f4',
        display: 'flex', flexDirection: 'column', overflow: 'hidden',
      }}>
        <div style={{ padding: '16px 16px 8px', borderBottom: '1px solid #313244' }}>
          <h3 style={{ margin: 0, color: '#89b4fa' }}>Route Optimizer</h3>
        </div>

        {/* controls */}
        <div style={{ padding: 16, borderBottom: '1px solid #313244' }}>
          <label style={{ fontSize: 13 }}>New HMB Latitude</label>
          <input value={lat} onChange={e => setLat(e.target.value)}
            style={inputStyle} />
          <label style={{ fontSize: 13 }}>New HMB Longitude</label>
          <input value={lon} onChange={e => setLon(e.target.value)}
            style={inputStyle} />
          <label style={{ fontSize: 13 }}>Run Type</label>
          <select value={runType} onChange={e => setRunType(e.target.value)}
            style={inputStyle}>
            <option value="both">Both (Insertion + Full)</option>
            <option value="insertion">Insertion Only</option>
            <option value="full">Full Re-Optimization Only</option>
          </select>

          <button onClick={handleRun} disabled={status === 'running'}
            style={{
              width: '100%', padding: '10px', marginTop: 8,
              background: status === 'running' ? '#585b70' : '#89b4fa',
              color: '#1e1e2e', border: 'none', borderRadius: 6,
              fontWeight: 600, fontSize: 14, cursor: status === 'running' ? 'wait' : 'pointer',
            }}>
            {status === 'running' ? `Running… (${phase})` : 'Run Optimization'}
          </button>
        </div>

        {/* results summary */}
        {fullResults && (
          <div style={{ padding: '12px 16px', borderBottom: '1px solid #313244', fontSize: 13 }}>
            <div style={{ color: '#a6e3a1', fontWeight: 600, marginBottom: 4 }}>Optimization Complete</div>
            <div>Original: <b>{fullResults.original_km} KM</b></div>
            <div>Optimized: <b>{fullResults.optimized_km} KM</b></div>
            <div style={{ color: '#a6e3a1' }}>Saved: <b>{fullResults.saving_km} KM</b></div>
          </div>
        )}

        {insertionResults && (
          <div style={{ padding: '12px 16px', borderBottom: '1px solid #313244', fontSize: 13 }}>
            <div style={{ color: '#fab387', fontWeight: 600, marginBottom: 4 }}>
              Best Insertion: Route {insertionResults[0]?.route_code}
            </div>
            <div>Extra KM: +{insertionResults[0]?.extra_km}</div>
            <div>Between: {insertionResults[0]?.prev_stop} → NEW → {insertionResults[0]?.next_stop}</div>
          </div>
        )}

        {/* live log */}
        <div ref={logRef} style={{
          flex: 1, overflow: 'auto', padding: '8px 12px',
          fontFamily: '"SF Mono", "Fira Code", monospace', fontSize: 11,
          lineHeight: 1.6, background: '#181825',
        }}>
          {logs.length === 0 && <span style={{ color: '#6c7086' }}>Logs will appear here…</span>}
          {logs.map((l, i) => (
            <div key={i} style={{ color: l.includes('ERROR') ? '#f38ba8' : '#a6adc8' }}>{l}</div>
          ))}
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
              <Popup><b>CC (Collection Center)</b></Popup>
            </Marker>
          )}

          {/* New HMB marker */}
          {newHmb && (
            <CircleMarker center={[newHmb.lat, newHmb.lng]} radius={10}
              pathOptions={{ color: '#f38ba8', fillColor: '#f38ba8', fillOpacity: 0.9 }}>
              <Popup><b>★ NEW HMB</b></Popup>
            </CircleMarker>
          )}

          {/* Route polylines + HMB markers */}
          {displayRoutes.map((route, idx) => {
            if (!route.hmbs || route.hmbs.length === 0) return null;
            const color = ROUTE_COLORS[idx % ROUTE_COLORS.length];
            const points = [];
            if (cc) points.push([cc.lat, cc.lng]);
            route.hmbs.forEach(h => points.push([h.lat, h.lng]));
            if (cc) points.push([cc.lat, cc.lng]);

            return (
              <React.Fragment key={route.route_code}>
                <Polyline positions={points}
                  pathOptions={{ color, weight: 3, opacity: 0.8 }} />
                {route.hmbs.map((h, hi) => (
                  <CircleMarker key={hi} center={[h.lat, h.lng]} radius={5}
                    pathOptions={{ color, fillColor: color, fillOpacity: 0.8 }}>
                    <Popup>
                      <b>{h.name}</b><br />
                      Route: {route.route_code} ({route.route_name})
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