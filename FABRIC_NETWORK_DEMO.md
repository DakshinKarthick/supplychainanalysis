# Fabric Network Demo — `~/projects/fabric-network`

> **Scope:** This guide is exclusively for the standalone Hyperledger Fabric network
> project at `~/projects/fabric-network` — the standard `test-network` with the
> `asset-transfer-basic` TypeScript chaincode.  
> It has **nothing to do** with the Supply Chain Analysis application in this repo.

---

## What this project is

`~/projects/fabric-network` is a local copy of the Hyperledger Fabric
**test-network** tutorial environment from [`hyperledger/fabric-samples`](https://github.com/hyperledger/fabric-samples).

It runs:

- **Two peer organisations** — Org1 (port 7051) and Org2 (port 9051)
- **One orderer** — `orderer.example.com` (port 7050)
- **Certificate Authorities** for each org
- **Channel:** `mychannel`
- **Chaincode:** `asset-transfer-basic` (TypeScript) — CRUD for generic assets

---

## Prerequisites (one-time)

```bash
# Docker must be running
sudo systemctl start docker   # or: open Docker Desktop on macOS/Windows
docker info                   # should print engine info with no error

# Node.js ≥ 18 and peer binary must be accessible
node --version                # v18.x.x or higher
peer version                  # shows Fabric 2.5.x
```

If `peer` is not found:

```bash
export PATH="${HOME}/fabric-samples/bin:${PATH}"
export FABRIC_CFG_PATH="${HOME}/fabric-samples/config"
```

Add both lines to `~/.bashrc` (or `~/.zshrc`) to make them permanent.

---

## Step-by-step transcript

### 1 — Navigate to the project

```bash
cd ~/projects/fabric-network
ls
# Expected contents:
#   network.sh  organizations/  scripts/  docker/  ...
```

---

### 2 — Clean up any previous run

Always start fresh to avoid stale containers or crypto material conflicts.

```bash
./network.sh down
```

**Expected output:**

```
Stopping network
Removing generated artifacts...
Removing docker volumes...
```

---

### 3 — Start the network and create a channel

```bash
./network.sh up createChannel -c mychannel
```

This single command:
1. Generates crypto material for all orgs
2. Starts orderer + both peers + CAs in Docker
3. Creates the channel `mychannel` and joins both peers

**Expected tail of output:**

```
Creating channel 'mychannel'
...
Channel 'mychannel' joined
```

Verify containers are running:

```bash
docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"
```

**Expected:**

```
NAMES                              STATUS          PORTS
peer0.org2.example.com             Up X seconds    0.0.0.0:9051->9051/tcp
peer0.org1.example.com             Up X seconds    0.0.0.0:7051->7051/tcp
orderer.example.com                Up X seconds    0.0.0.0:7050->7050/tcp
ca_org2                            Up X seconds    0.0.0.0:8054->8054/tcp
ca_org1                            Up X seconds    0.0.0.0:7054->7054/tcp
ca_orderer                         Up X seconds    0.0.0.0:9054->9054/tcp
```

---

### 4 — Deploy the chaincode

```bash
./network.sh deployCC \
  -ccn basic \
  -ccp ../asset-transfer-basic/chaincode-typescript \
  -ccl typescript \
  -c  mychannel
```

This will:
1. Install the chaincode on both peers
2. Approve the chaincode definition for both orgs
3. Commit the chaincode to `mychannel`

**Expected tail:**

```
Chaincode definition committed on channel 'mychannel'
```

---

### 5 — Set up the CLI environment

Run these in every new terminal session before using `peer` commands:

```bash
export PATH="${HOME}/fabric-samples/bin:${PATH}"
export FABRIC_CFG_PATH="${HOME}/fabric-samples/config"
export CORE_PEER_TLS_ENABLED=true

# Switch to Org1
export CORE_PEER_LOCALMSPID="Org1MSP"
export CORE_PEER_TLS_ROOTCERT_FILE="${HOME}/projects/fabric-network/organizations/peerOrganizations/org1.example.com/peers/peer0.org1.example.com/tls/ca.crt"
export CORE_PEER_MSPCONFIGPATH="${HOME}/projects/fabric-network/organizations/peerOrganizations/org1.example.com/users/Admin@org1.example.com/msp"
export CORE_PEER_ADDRESS=localhost:7051

# Shared variables for invoke commands
export ORDERER_CA="${HOME}/projects/fabric-network/organizations/ordererOrganizations/example.com/orderers/orderer.example.com/msp/tlscacerts/tlsca.example.com-cert.pem"
export ORG1_TLS="${HOME}/projects/fabric-network/organizations/peerOrganizations/org1.example.com/peers/peer0.org1.example.com/tls/ca.crt"
export ORG2_TLS="${HOME}/projects/fabric-network/organizations/peerOrganizations/org2.example.com/peers/peer0.org2.example.com/tls/ca.crt"
```

---

### 6 — Initialise the ledger

```bash
peer chaincode invoke \
  -o localhost:7050 \
  --ordererTLSHostnameOverride orderer.example.com \
  --tls --cafile "${ORDERER_CA}" \
  -C mychannel -n basic \
  --peerAddresses localhost:7051 --tlsRootCertFiles "${ORG1_TLS}" \
  --peerAddresses localhost:9051 --tlsRootCertFiles "${ORG2_TLS}" \
  -c '{"function":"InitLedger","Args":[]}'
```

**Expected:**

```
Chaincode invoke successful. result: status:200
```

Wait 3 seconds for the block to commit, then proceed.

---

### 7 — Query all assets

```bash
peer chaincode query \
  -C mychannel -n basic \
  -c '{"function":"GetAllAssets","Args":[]}' \
  | python3 -m json.tool
```

**Expected output:**

```json
[
  { "ID": "asset1", "Color": "blue",   "Size": 5,  "Owner": "Tomoko",   "AppraisedValue": 300 },
  { "ID": "asset2", "Color": "red",    "Size": 5,  "Owner": "Brad",     "AppraisedValue": 400 },
  { "ID": "asset3", "Color": "green",  "Size": 10, "Owner": "Jin Soo",  "AppraisedValue": 500 },
  { "ID": "asset4", "Color": "yellow", "Size": 10, "Owner": "Max",      "AppraisedValue": 600 },
  { "ID": "asset5", "Color": "black",  "Size": 15, "Owner": "Adriana",  "AppraisedValue": 700 },
  { "ID": "asset6", "Color": "white",  "Size": 15, "Owner": "Michel",   "AppraisedValue": 800 }
]
```

---

### 8 — Query a single asset

```bash
peer chaincode query \
  -C mychannel -n basic \
  -c '{"function":"ReadAsset","Args":["asset1"]}' \
  | python3 -m json.tool
```

**Expected:**

```json
{ "ID": "asset1", "Color": "blue", "Size": 5, "Owner": "Tomoko", "AppraisedValue": 300 }
```

---

### 9 — Create a new asset

```bash
peer chaincode invoke \
  -o localhost:7050 \
  --ordererTLSHostnameOverride orderer.example.com \
  --tls --cafile "${ORDERER_CA}" \
  -C mychannel -n basic \
  --peerAddresses localhost:7051 --tlsRootCertFiles "${ORG1_TLS}" \
  --peerAddresses localhost:9051 --tlsRootCertFiles "${ORG2_TLS}" \
  -c '{"function":"CreateAsset","Args":["asset7","purple",20,"Karthick",1000]}'
```

**Expected:**

```
Chaincode invoke successful. result: status:200
```

Verify it was created:

```bash
peer chaincode query \
  -C mychannel -n basic \
  -c '{"function":"ReadAsset","Args":["asset7"]}' \
  | python3 -m json.tool
```

```json
{ "ID": "asset7", "Color": "purple", "Size": 20, "Owner": "Karthick", "AppraisedValue": 1000 }
```

---

### 10 — Update an asset

```bash
peer chaincode invoke \
  -o localhost:7050 \
  --ordererTLSHostnameOverride orderer.example.com \
  --tls --cafile "${ORDERER_CA}" \
  -C mychannel -n basic \
  --peerAddresses localhost:7051 --tlsRootCertFiles "${ORG1_TLS}" \
  --peerAddresses localhost:9051 --tlsRootCertFiles "${ORG2_TLS}" \
  -c '{"function":"UpdateAsset","Args":["asset7","purple",25,"Karthick",1200]}'
```

Verify the update:

```bash
peer chaincode query \
  -C mychannel -n basic \
  -c '{"function":"ReadAsset","Args":["asset7"]}' \
  | python3 -m json.tool
```

```json
{ "ID": "asset7", "Color": "purple", "Size": 25, "Owner": "Karthick", "AppraisedValue": 1200 }
```

---

### 11 — Transfer asset ownership

```bash
peer chaincode invoke \
  -o localhost:7050 \
  --ordererTLSHostnameOverride orderer.example.com \
  --tls --cafile "${ORDERER_CA}" \
  -C mychannel -n basic \
  --peerAddresses localhost:7051 --tlsRootCertFiles "${ORG1_TLS}" \
  --peerAddresses localhost:9051 --tlsRootCertFiles "${ORG2_TLS}" \
  -c '{"function":"TransferAsset","Args":["asset7","Dakshin"]}'
```

Verify:

```bash
peer chaincode query \
  -C mychannel -n basic \
  -c '{"function":"ReadAsset","Args":["asset7"]}' \
  | python3 -m json.tool
```

```json
{ "ID": "asset7", "Color": "purple", "Size": 25, "Owner": "Dakshin", "AppraisedValue": 1200 }
```

---

### 12 — Cross-organisation verification (Org2)

Demonstrates that both organisations share the same ledger state — a core
property of a permissioned blockchain.

```bash
# Switch to Org2
export CORE_PEER_LOCALMSPID="Org2MSP"
export CORE_PEER_TLS_ROOTCERT_FILE="${HOME}/projects/fabric-network/organizations/peerOrganizations/org2.example.com/peers/peer0.org2.example.com/tls/ca.crt"
export CORE_PEER_MSPCONFIGPATH="${HOME}/projects/fabric-network/organizations/peerOrganizations/org2.example.com/users/Admin@org2.example.com/msp"
export CORE_PEER_ADDRESS=localhost:9051

# Query from Org2's peer — result must be identical
peer chaincode query \
  -C mychannel -n basic \
  -c '{"function":"ReadAsset","Args":["asset7"]}' \
  | python3 -m json.tool
```

**Expected (same data as Org1):**

```json
{ "ID": "asset7", "Color": "purple", "Size": 25, "Owner": "Dakshin", "AppraisedValue": 1200 }
```

This confirms **consensus** — both peers agreed on and committed the same block.

---

### 13 — Delete an asset

```bash
# Switch back to Org1 first
export CORE_PEER_LOCALMSPID="Org1MSP"
export CORE_PEER_TLS_ROOTCERT_FILE="${HOME}/projects/fabric-network/organizations/peerOrganizations/org1.example.com/peers/peer0.org1.example.com/tls/ca.crt"
export CORE_PEER_MSPCONFIGPATH="${HOME}/projects/fabric-network/organizations/peerOrganizations/org1.example.com/users/Admin@org1.example.com/msp"
export CORE_PEER_ADDRESS=localhost:7051

peer chaincode invoke \
  -o localhost:7050 \
  --ordererTLSHostnameOverride orderer.example.com \
  --tls --cafile "${ORDERER_CA}" \
  -C mychannel -n basic \
  --peerAddresses localhost:7051 --tlsRootCertFiles "${ORG1_TLS}" \
  --peerAddresses localhost:9051 --tlsRootCertFiles "${ORG2_TLS}" \
  -c '{"function":"DeleteAsset","Args":["asset7"]}'
```

Confirm it no longer exists:

```bash
peer chaincode query \
  -C mychannel -n basic \
  -c '{"function":"AssetExists","Args":["asset7"]}' \
  | python3 -m json.tool
```

**Expected:** `false`

---

### 14 — Tear down the network

```bash
cd ~/projects/fabric-network
./network.sh down
```

**Expected:**

```
Stopping network
Removing docker volumes...
```

Confirm all Fabric containers are gone:

```bash
docker ps | grep -E "peer|orderer|ca_"
# (no output)
```

---

## Quick-reference cheat sheet

```bash
# ── Navigate ─────────────────────────────────────────────────────────────────
cd ~/projects/fabric-network

# ── Network lifecycle ────────────────────────────────────────────────────────
./network.sh down                                                   # stop
./network.sh up createChannel -c mychannel                          # start + channel
./network.sh deployCC -ccn basic -ccp ../asset-transfer-basic/chaincode-typescript -ccl typescript -c mychannel  # deploy
docker ps                                                           # verify

# ── CLI environment (Org1) ───────────────────────────────────────────────────
export PATH="${HOME}/fabric-samples/bin:${PATH}"
export FABRIC_CFG_PATH="${HOME}/fabric-samples/config"
export CORE_PEER_TLS_ENABLED=true
export CORE_PEER_LOCALMSPID="Org1MSP"
FN="${HOME}/projects/fabric-network"
export CORE_PEER_TLS_ROOTCERT_FILE="${FN}/organizations/peerOrganizations/org1.example.com/peers/peer0.org1.example.com/tls/ca.crt"
export CORE_PEER_MSPCONFIGPATH="${FN}/organizations/peerOrganizations/org1.example.com/users/Admin@org1.example.com/msp"
export CORE_PEER_ADDRESS=localhost:7051
export ORDERER_CA="${FN}/organizations/ordererOrganizations/example.com/orderers/orderer.example.com/msp/tlscacerts/tlsca.example.com-cert.pem"
export ORG1_TLS="${FN}/organizations/peerOrganizations/org1.example.com/peers/peer0.org1.example.com/tls/ca.crt"
export ORG2_TLS="${FN}/organizations/peerOrganizations/org2.example.com/peers/peer0.org2.example.com/tls/ca.crt"

INVOKE="peer chaincode invoke -o localhost:7050 --ordererTLSHostnameOverride orderer.example.com --tls --cafile ${ORDERER_CA} -C mychannel -n basic --peerAddresses localhost:7051 --tlsRootCertFiles ${ORG1_TLS} --peerAddresses localhost:9051 --tlsRootCertFiles ${ORG2_TLS}"
QUERY="peer chaincode query -C mychannel -n basic"

# ── Read operations ──────────────────────────────────────────────────────────
${QUERY} -c '{"function":"GetAllAssets","Args":[]}'                 | python3 -m json.tool
${QUERY} -c '{"function":"ReadAsset","Args":["asset1"]}'            | python3 -m json.tool
${QUERY} -c '{"function":"AssetExists","Args":["asset7"]}'          | python3 -m json.tool

# ── Write operations ─────────────────────────────────────────────────────────
${INVOKE} -c '{"function":"InitLedger","Args":[]}'
${INVOKE} -c '{"function":"CreateAsset","Args":["asset7","purple",20,"Karthick",1000]}'
${INVOKE} -c '{"function":"UpdateAsset","Args":["asset7","purple",25,"Karthick",1200]}'
${INVOKE} -c '{"function":"TransferAsset","Args":["asset7","Dakshin"]}'
${INVOKE} -c '{"function":"DeleteAsset","Args":["asset7"]}'
```

---

## Troubleshooting

### `Error: endorsement failure during invoke`
Make sure both `--peerAddresses` (Org1 + Org2) are passed on every invoke — the default endorsement policy requires sign-off from both organisations.

### Containers already running / port conflict
```bash
./network.sh down
docker ps   # should be empty of Fabric containers
./network.sh up createChannel -c mychannel
```

### `peer: command not found`
```bash
export PATH="${HOME}/fabric-samples/bin:${PATH}"
```

### Chaincode times out during `deployCC`
Docker may still be pulling chaincode builder images.  Wait 2–3 minutes and re-run `deployCC`.

### `GetHistoryForKey` not available
Transaction history queries require CouchDB state database.  Start the network with:
```bash
./network.sh up createChannel -c mychannel -s couchdb
```
