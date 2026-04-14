# Supply Chain Analysis — Hyperledger Fabric Demo Guide

This guide covers **everything** you need to run and showcase the project's
Hyperledger Fabric chaincode component to an audience.  It assumes a fresh
Ubuntu / Debian machine, but the commands translate directly to macOS or WSL2.

---

## Table of Contents

1. [Architecture Overview](#1-architecture-overview)
2. [Prerequisites](#2-prerequisites)
3. [Install Docker](#3-install-docker)
4. [Download Fabric Binaries & Test Network](#4-download-fabric-binaries--test-network)
5. [Build the Chaincode](#5-build-the-chaincode)
6. [Start the Fabric Network & Deploy Chaincode](#6-start-the-fabric-network--deploy-chaincode)
7. [Interact with the Chaincode (Full Transcript)](#7-interact-with-the-chaincode-full-transcript)
8. [Run the Python Optimizer + Write to Blockchain](#8-run-the-python-optimizer--write-to-blockchain)
9. [Tear Down](#9-tear-down)
10. [Troubleshooting](#10-troubleshooting)
11. [Quick Reference — Command Cheat Sheet](#11-quick-reference--command-cheat-sheet)

---

## 1. Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                  Supply Chain Analysis System                    │
│                                                                 │
│  ┌──────────────┐   optimizer result   ┌─────────────────────┐  │
│  │ Python       │ ──────────────────►  │ Hyperledger Fabric  │  │
│  │ Route        │                      │ (blockchain ledger) │  │
│  │ Optimizer    │ ◄─────────────────── │                     │  │
│  └──────────────┘   query routes       └─────────────────────┘  │
│         ▲                                        │              │
│         │ GPS + milk qty                         │ immutable    │
│  ┌──────┴───────┐                      ┌─────────▼──────────┐  │
│  │  React/Tauri │                      │   Channel:          │  │
│  │  Frontend    │                      │   mychannel         │  │
│  └──────────────┘                      │                     │  │
│                                        │  Chaincode:         │  │
│                                        │  supply-chain v1.0  │  │
│                                        └─────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

**What lives on the blockchain:**

| Asset | Key format | Description |
|-------|-----------|-------------|
| `CollectionRoute` | `ROUTE-XX` | Route metadata, vehicle, max capacity, stops |
| `HMBAssignment` | `ASSIGN-XXX` | Optimizer decision: which HMB goes to which route |

**Chaincode functions:**

| Function | Type | Description |
|----------|------|-------------|
| `InitLedger` | Invoke | Seeds ledger with 7 existing routes |
| `CreateRoute` | Invoke | Admin: add a new route |
| `QueryRoute` | Query | Fetch one route by ID |
| `QueryAllRoutes` | Query | List all routes |
| `RecordHMBAssignment` | Invoke | Write optimizer decision to ledger |
| `QueryAssignment` | Query | Fetch one assignment by ID |
| `GetAssignmentHistory` | Query | Full blockchain history for any key |
| `UpdateRouteLoad` | Invoke | Admin: manually correct route load |

---

## 2. Prerequisites

| Tool | Minimum version | Check |
|------|----------------|-------|
| Docker + Compose | 24.x + Compose v2 | `docker --version && docker compose version` |
| Node.js | 18.x | `node --version` |
| Git | any | `git --version` |
| curl | any | `curl --version` |
| Python 3 | 3.9+ | `python3 --version` *(for optimizer integration)* |

---

## 3. Install Docker

> Skip if `docker info` returns without error.

```bash
# Ubuntu / Debian
sudo apt-get update
sudo apt-get install -y ca-certificates curl gnupg lsb-release

sudo install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg \
  | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
sudo chmod a+r /etc/apt/keyrings/docker.gpg

echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] \
  https://download.docker.com/linux/ubuntu \
  $(lsb_release -cs) stable" \
  | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

sudo apt-get update
sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

# Allow running docker without sudo (log out and back in after this)
sudo usermod -aG docker "$USER"
newgrp docker

# Verify
docker run --rm hello-world
docker --version
docker compose version
```

**Expected output (example):**

```
Docker version 24.0.5, build ced0996
Docker Compose version v2.20.3
```

---

## 4. Download Fabric Binaries & Test Network

```bash
# Download fabric-samples, peer/orderer binaries, and Docker images
# (Fabric 2.5.15 + CA 1.5.15 — versions validated against this project)
curl -sSL https://bit.ly/2ysbOFE | bash -s -- 2.5.15 1.5.15 -d -s

# This creates ~/fabric-samples/
ls ~/fabric-samples/
# Expected: asset-transfer-basic  bin  config  test-network  ...

# Add the Fabric binaries to PATH for this session
export PATH="${HOME}/fabric-samples/bin:${PATH}"
export FABRIC_CFG_PATH="${HOME}/fabric-samples/config"

# Verify
peer version
# Expected: peer:  Version: 2.5.15  Commit SHA: ...
```

**Pulling images takes 3–10 minutes on first run.**  If you hit timeouts, run:

```bash
docker pull hyperledger/fabric-peer:2.5.15
docker pull hyperledger/fabric-orderer:2.5.15
docker pull hyperledger/fabric-tools:2.5.15
docker pull hyperledger/fabric-ca:1.5.15
```

---

## 5. Build the Chaincode

```bash
cd chaincode/supply-chain
npm install
npm run build

# Expected:
# added 42 packages in 8s
# > supply-chain-chaincode@1.0.0 build
# > tsc
ls dist/
# Expected: index.d.ts  index.js  src/
```

---

## 6. Start the Fabric Network & Deploy Chaincode

### Option A — Automated (recommended)

```bash
# From the repo root
bash fabric/setup.sh
```

The script will:
1. Start the two-org test network with TLS enabled
2. Create channel `mychannel`
3. Deploy the `supply-chain` chaincode
4. Call `InitLedger` to seed the ledger with 7 routes
5. Run a quick `QueryAllRoutes` smoke test

### Option B — Manual step by step

```bash
cd ~/fabric-samples/test-network

# Step 1: Clean any previous state
./network.sh down

# Step 2: Start network + create channel
./network.sh up createChannel -c mychannel -ca

# Expected tail of output:
# Channel 'mychannel' joined

# Step 3: Deploy chaincode
./network.sh deployCC \
  -ccn supply-chain \
  -ccp /path/to/supplychainanalysis/chaincode/supply-chain \
  -ccl typescript \
  -c  mychannel

# Expected tail:
# Chaincode definition committed on channel 'mychannel'
```

### Verify the network is running

```bash
docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"
```

**Expected output:**

```
NAMES                                        STATUS          PORTS
peer0.org2.example.com                       Up 2 minutes    0.0.0.0:9051->9051/tcp
peer0.org1.example.com                       Up 2 minutes    0.0.0.0:7051->7051/tcp
orderer.example.com                          Up 2 minutes    0.0.0.0:7050->7050/tcp
ca_org2                                      Up 2 minutes    0.0.0.0:8054->8054/tcp
ca_org1                                      Up 2 minutes    0.0.0.0:7054->7054/tcp
ca_orderer                                   Up 2 minutes    0.0.0.0:9054->9054/tcp
```

---

## 7. Interact with the Chaincode (Full Transcript)

Set up the environment variables for all commands below:

```bash
export PATH="${HOME}/fabric-samples/bin:${PATH}"
export FABRIC_CFG_PATH="${HOME}/fabric-samples/config"
export CORE_PEER_TLS_ENABLED=true
export CORE_PEER_LOCALMSPID="Org1MSP"
export CORE_PEER_TLS_ROOTCERT_FILE="${HOME}/fabric-samples/test-network/organizations/peerOrganizations/org1.example.com/peers/peer0.org1.example.com/tls/ca.crt"
export CORE_PEER_MSPCONFIGPATH="${HOME}/fabric-samples/test-network/organizations/peerOrganizations/org1.example.com/users/Admin@org1.example.com/msp"
export CORE_PEER_ADDRESS=localhost:7051
export ORDERER_CA="${HOME}/fabric-samples/test-network/organizations/ordererOrganizations/example.com/orderers/orderer.example.com/msp/tlscacerts/tlsca.example.com-cert.pem"
export ORG1_TLS="${HOME}/fabric-samples/test-network/organizations/peerOrganizations/org1.example.com/peers/peer0.org1.example.com/tls/ca.crt"
export ORG2_TLS="${HOME}/fabric-samples/test-network/organizations/peerOrganizations/org2.example.com/peers/peer0.org2.example.com/tls/ca.crt"
```

---

### 7.1 Initialise the Ledger

```bash
peer chaincode invoke \
  -o localhost:7050 \
  --ordererTLSHostnameOverride orderer.example.com \
  --tls --cafile "${ORDERER_CA}" \
  -C mychannel -n supply-chain \
  --peerAddresses localhost:7051 --tlsRootCertFiles "${ORG1_TLS}" \
  --peerAddresses localhost:9051 --tlsRootCertFiles "${ORG2_TLS}" \
  -c '{"function":"InitLedger","Args":[]}'
```

**Expected:**

```
2024-xx-xx ... INFO [chaincodeCmd] chaincodeInvokeOrQuery -> Chaincode invoke successful. result: status:200
```

---

### 7.2 Query All Routes

```bash
peer chaincode query \
  -C mychannel -n supply-chain \
  -c '{"function":"QueryAllRoutes","Args":[]}' \
  | python3 -m json.tool
```

**Expected output (truncated):**

```json
[
  {
    "docType": "collectionRoute",
    "routeId": "ROUTE-01",
    "plantId": "1142",
    "plantName": "Uthangarai CC",
    "vehicleType": "Tanker",
    "maxCapacityLitres": 2000,
    "currentLoadLitres": 1450,
    "stops": [
      { "hmbId": "HMB-101", "name": "Uthangarai", "lat": 12.308573, "lon": 78.535901, "expectedMilkQty": 200 },
      { "hmbId": "HMB-102", "name": "Chinnakaruppur", "lat": 12.33, "lon": 78.51, "expectedMilkQty": 250 },
      { "hmbId": "HMB-103", "name": "Kottaiyur", "lat": 12.36, "lon": 78.49, "expectedMilkQty": 300 }
    ],
    "createdAt": "2024-05-01T10:00:00.000Z",
    "updatedAt": "2024-05-01T10:00:00.000Z"
  },
  ...  (7 routes total)
]
```

---

### 7.3 Query a Specific Route

```bash
peer chaincode query \
  -C mychannel -n supply-chain \
  -c '{"function":"QueryRoute","Args":["ROUTE-04"]}' \
  | python3 -m json.tool
```

**Expected:**

```json
{
  "docType": "collectionRoute",
  "routeId": "ROUTE-04",
  "vehicleType": "Tanker",
  "maxCapacityLitres": 2000,
  "currentLoadLitres": 1300,
  "stops": [
    { "hmbId": "HMB-401", "name": "Morappur", ... },
    { "hmbId": "HMB-402", "name": "Odaikadu", ... },
    { "hmbId": "HMB-403", "name": "Periyakurumbadi", ... },
    { "hmbId": "HMB-404", "name": "Kuppandapadi", ... },
    { "hmbId": "HMB-405", "name": "Vellariveli", ... }
  ]
}
```

---

### 7.4 Record an HMB Assignment (Optimizer → Blockchain)

This is the core integration: the Python optimizer decides the best route for a
new HMB, then the result is **permanently recorded** on the blockchain.

```bash
peer chaincode invoke \
  -o localhost:7050 \
  --ordererTLSHostnameOverride orderer.example.com \
  --tls --cafile "${ORDERER_CA}" \
  -C mychannel -n supply-chain \
  --peerAddresses localhost:7051 --tlsRootCertFiles "${ORG1_TLS}" \
  --peerAddresses localhost:9051 --tlsRootCertFiles "${ORG2_TLS}" \
  -c '{
    "function": "RecordHMBAssignment",
    "Args": [
      "ASSIGN-001",
      "HMB-NEW-01",
      "Thoppur Village",
      "12.35",
      "78.55",
      "100",
      "ROUTE-04",
      "3",
      "2.4",
      "95.0",
      "0.312",
      "SupplyChainOptimizer-v2"
    ]
  }'
```

**Arguments explained:**

| Position | Value | Meaning |
|----------|-------|---------|
| 1 | `ASSIGN-001` | Unique assignment ID |
| 2 | `HMB-NEW-01` | New HMB identifier |
| 3 | `Thoppur Village` | HMB location name |
| 4 | `12.35` | Latitude |
| 5 | `78.55` | Longitude |
| 6 | `100` | Expected milk qty (litres) |
| 7 | `ROUTE-04` | Route assigned by optimizer |
| 8 | `3` | Insertion position in stop list |
| 9 | `2.4` | Extra distance added (km) |
| 10 | `95.0` | Total route km after insertion |
| 11 | `0.312` | Optimizer composite score |
| 12 | `SupplyChainOptimizer-v2` | System / operator identity |

**Expected:**

```
Chaincode invoke successful. result: status:200
```

---

### 7.5 Verify Assignment Record

```bash
peer chaincode query \
  -C mychannel -n supply-chain \
  -c '{"function":"QueryAssignment","Args":["ASSIGN-001"]}' \
  | python3 -m json.tool
```

**Expected:**

```json
{
  "docType": "hmbAssignment",
  "assignmentId": "ASSIGN-001",
  "hmbId": "HMB-NEW-01",
  "hmbName": "Thoppur Village",
  "lat": 12.35,
  "lon": 78.55,
  "expectedMilkQty": 100,
  "assignedRouteId": "ROUTE-04",
  "insertionPosition": 3,
  "extraDistanceKm": 2.4,
  "totalRouteKmAfter": 95.0,
  "optimizerScore": 0.312,
  "assignedAt": "2024-05-01T10:05:00.000Z",
  "assignedBy": "SupplyChainOptimizer-v2"
}
```

---

### 7.6 Verify Route was Updated

After the assignment, `ROUTE-04` should show:
- `currentLoadLitres` increased by 100 (from 1300 → 1400)
- New stop `HMB-NEW-01` inserted at position 3

```bash
peer chaincode query \
  -C mychannel -n supply-chain \
  -c '{"function":"QueryRoute","Args":["ROUTE-04"]}' \
  | python3 -m json.tool
```

---

### 7.7 View Full Blockchain History

The `GetAssignmentHistory` function shows every version of a key, with
transaction IDs — demonstrating the **immutable audit trail** on the blockchain.

```bash
peer chaincode query \
  -C mychannel -n supply-chain \
  -c '{"function":"GetAssignmentHistory","Args":["ROUTE-04"]}' \
  | python3 -m json.tool
```

**Expected:**

```json
[
  {
    "txId": "a1b2c3d4...",
    "timestamp": { "seconds": "1714560000", "nanos": 0 },
    "isDelete": false,
    "data": { "routeId": "ROUTE-04", "currentLoadLitres": 1300, ... }
  },
  {
    "txId": "e5f6a7b8...",
    "timestamp": { "seconds": "1714560300", "nanos": 0 },
    "isDelete": false,
    "data": { "routeId": "ROUTE-04", "currentLoadLitres": 1400, ... }
  }
]
```

---

### 7.8 Create a New Route

```bash
peer chaincode invoke \
  -o localhost:7050 \
  --ordererTLSHostnameOverride orderer.example.com \
  --tls --cafile "${ORDERER_CA}" \
  -C mychannel -n supply-chain \
  --peerAddresses localhost:7051 --tlsRootCertFiles "${ORG1_TLS}" \
  --peerAddresses localhost:9051 --tlsRootCertFiles "${ORG2_TLS}" \
  -c '{"function":"CreateRoute","Args":["ROUTE-08","1142","Uthangarai CC","Tanker","2000"]}'
```

---

### 7.9 Update Route Load

```bash
peer chaincode invoke \
  -o localhost:7050 \
  --ordererTLSHostnameOverride orderer.example.com \
  --tls --cafile "${ORDERER_CA}" \
  -C mychannel -n supply-chain \
  --peerAddresses localhost:7051 --tlsRootCertFiles "${ORG1_TLS}" \
  --peerAddresses localhost:9051 --tlsRootCertFiles "${ORG2_TLS}" \
  -c '{"function":"UpdateRouteLoad","Args":["ROUTE-08","500"]}'
```

---

### 7.10 Cross-Organisation Verification (Org2)

Switch to Org2's identity and query — demonstrating that both organisations see
identical ledger state:

```bash
export CORE_PEER_LOCALMSPID="Org2MSP"
export CORE_PEER_TLS_ROOTCERT_FILE="${HOME}/fabric-samples/test-network/organizations/peerOrganizations/org2.example.com/peers/peer0.org2.example.com/tls/ca.crt"
export CORE_PEER_MSPCONFIGPATH="${HOME}/fabric-samples/test-network/organizations/peerOrganizations/org2.example.com/users/Admin@org2.example.com/msp"
export CORE_PEER_ADDRESS=localhost:9051

peer chaincode query \
  -C mychannel -n supply-chain \
  -c '{"function":"QueryRoute","Args":["ROUTE-04"]}' \
  | python3 -m json.tool
```

**Result is identical to Org1's view** — verifying cross-org consensus.

---

## 8. Run the Python Optimizer + Write to Blockchain

This is the end-to-end integration: run the existing route optimizer and
automatically push its recommendation to the Fabric ledger.

```bash
# 1. Run the optimizer
python3 script/route_optimizer.py --lat 12.35 --lon 78.55 --milk-qty 100

# Note the recommended route (e.g., ROUTE-04) and optimizer score from the output

# 2. Record the recommendation on-chain
# Replace <ROUTE_ID>, <SCORE>, <EXTRA_KM>, <TOTAL_KM> with actual optimizer output
peer chaincode invoke \
  -o localhost:7050 \
  --ordererTLSHostnameOverride orderer.example.com \
  --tls --cafile "${ORDERER_CA}" \
  -C mychannel -n supply-chain \
  --peerAddresses localhost:7051 --tlsRootCertFiles "${ORG1_TLS}" \
  --peerAddresses localhost:9051 --tlsRootCertFiles "${ORG2_TLS}" \
  -c '{
    "function":"RecordHMBAssignment",
    "Args":["ASSIGN-002","HMB-TEST","New Test Village","12.35","78.55","100",
            "<ROUTE_ID>","<POSITION>","<EXTRA_KM>","<TOTAL_KM>","<SCORE>",
            "demo-operator"]
  }'

# 3. Verify it is on the ledger
peer chaincode query \
  -C mychannel -n supply-chain \
  -c '{"function":"QueryAssignment","Args":["ASSIGN-002"]}' \
  | python3 -m json.tool
```

---

## 9. Tear Down

```bash
# Stop the network (removes containers, but NOT the crypto material)
cd ~/fabric-samples/test-network
./network.sh down

# Or use the helper script
bash fabric/setup.sh --down

# Verify all Fabric containers are gone
docker ps | grep -E "peer|orderer|ca_"
```

---

## 10. Troubleshooting

### Docker pull timeouts

```bash
# Pull images manually one by one
for img in peer orderer tools ca; do
  docker pull hyperledger/fabric-${img}:2.5.15
done
docker pull hyperledger/fabric-ca:1.5.15
```

### `peer: command not found`

```bash
export PATH="${HOME}/fabric-samples/bin:${PATH}"
```

Add this line to `~/.bashrc` to make it permanent.

### Chaincode commit fails with endorsement error

Make sure you pass both `--peerAddresses` flags (Org1 + Org2) in all invoke
commands.  A single-org endorsement will be rejected by the default policy.

### `Error: could not assemble transaction: ProposalResponsePayloads do not match`

This usually means TLS certificates are mismatched.  Re-check that each
`--tlsRootCertFiles` path matches the corresponding `--peerAddresses` org.

### How to re-deploy after code changes

```bash
cd ~/fabric-samples/test-network
./network.sh down
# Bump the version in chaincode/supply-chain/package.json, then:
bash /path/to/supplychainanalysis/fabric/setup.sh
```

---

## 11. Quick Reference — Command Cheat Sheet

```bash
# ── Environment (run once per terminal session) ──────────────────────────────
export PATH="${HOME}/fabric-samples/bin:${PATH}"
export FABRIC_CFG_PATH="${HOME}/fabric-samples/config"
export CORE_PEER_TLS_ENABLED=true
export CORE_PEER_LOCALMSPID="Org1MSP"
TN="${HOME}/fabric-samples/test-network"
export CORE_PEER_TLS_ROOTCERT_FILE="${TN}/organizations/peerOrganizations/org1.example.com/peers/peer0.org1.example.com/tls/ca.crt"
export CORE_PEER_MSPCONFIGPATH="${TN}/organizations/peerOrganizations/org1.example.com/users/Admin@org1.example.com/msp"
export CORE_PEER_ADDRESS=localhost:7051
export ORDERER_CA="${TN}/organizations/ordererOrganizations/example.com/orderers/orderer.example.com/msp/tlscacerts/tlsca.example.com-cert.pem"
export ORG1_TLS="${TN}/organizations/peerOrganizations/org1.example.com/peers/peer0.org1.example.com/tls/ca.crt"
export ORG2_TLS="${TN}/organizations/peerOrganizations/org2.example.com/peers/peer0.org2.example.com/tls/ca.crt"

INVOKE="peer chaincode invoke -o localhost:7050 --ordererTLSHostnameOverride orderer.example.com --tls --cafile ${ORDERER_CA} -C mychannel -n supply-chain --peerAddresses localhost:7051 --tlsRootCertFiles ${ORG1_TLS} --peerAddresses localhost:9051 --tlsRootCertFiles ${ORG2_TLS}"
QUERY="peer chaincode query -C mychannel -n supply-chain"

# ── Read operations ──────────────────────────────────────────────────────────
${QUERY} -c '{"function":"QueryAllRoutes","Args":[]}'   | python3 -m json.tool
${QUERY} -c '{"function":"QueryRoute","Args":["ROUTE-04"]}' | python3 -m json.tool
${QUERY} -c '{"function":"QueryAssignment","Args":["ASSIGN-001"]}' | python3 -m json.tool
${QUERY} -c '{"function":"GetAssignmentHistory","Args":["ROUTE-04"]}' | python3 -m json.tool

# ── Write operations ─────────────────────────────────────────────────────────
${INVOKE} -c '{"function":"InitLedger","Args":[]}'
${INVOKE} -c '{"function":"CreateRoute","Args":["ROUTE-08","1142","Uthangarai CC","Tanker","2000"]}'
${INVOKE} -c '{"function":"UpdateRouteLoad","Args":["ROUTE-08","500"]}'
${INVOKE} -c '{"function":"RecordHMBAssignment","Args":["ASSIGN-001","HMB-NEW-01","Thoppur Village","12.35","78.55","100","ROUTE-04","3","2.4","95.0","0.312","demo-operator"]}'

# ── Network management ───────────────────────────────────────────────────────
docker ps                        # list running containers
cd ~/fabric-samples/test-network
./network.sh up createChannel -c mychannel -ca   # start
./network.sh down                                # stop
```
