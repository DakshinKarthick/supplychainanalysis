#!/usr/bin/env bash
# =============================================================================
# setup.sh — One-shot Fabric test-network setup for Supply Chain Analysis
#
# Usage:
#   bash fabric/setup.sh [--down]
#
# What it does:
#   1. Downloads the Hyperledger Fabric test-network (fabric-samples) if absent.
#   2. Starts the two-org test network + creates "mychannel".
#   3. Builds the TypeScript chaincode.
#   4. Deploys "supply-chain" chaincode to mychannel.
#   5. Initialises the ledger (seeds 7 routes).
#
# Prerequisites (install first):
#   • Docker  ≥ 24.x   — https://docs.docker.com/engine/install/ubuntu/
#   • Docker Compose v2 (bundled with modern Docker Desktop / Docker CE)
#   • Node.js ≥ 18.x   — https://nodejs.org/en/download/
#   • curl, git
#
# =============================================================================

set -euo pipefail

# ── Paths ────────────────────────────────────────────────────────────────────
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CHAINCODE_PATH="${REPO_ROOT}/chaincode/supply-chain"
FABRIC_SAMPLES_DIR="${HOME}/fabric-samples"
TEST_NETWORK="${FABRIC_SAMPLES_DIR}/test-network"

CHANNEL="mychannel"
CC_NAME="supply-chain"
CC_LABEL="${CC_NAME}_1.0"

# ── Colour helpers ────────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'
info()  { echo -e "${GREEN}[INFO]${NC}  $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC}  $*"; }
error() { echo -e "${RED}[ERROR]${NC} $*" >&2; exit 1; }

# ── Teardown flag ─────────────────────────────────────────────────────────────
if [[ "${1:-}" == "--down" ]]; then
  info "Tearing down the network …"
  cd "${TEST_NETWORK}"
  ./network.sh down
  info "Network stopped."
  exit 0
fi

# ── 1. Preflight checks ───────────────────────────────────────────────────────
info "Checking prerequisites …"

command -v docker >/dev/null 2>&1  || error "Docker not found. Install from https://docs.docker.com/engine/install/"
command -v node   >/dev/null 2>&1  || error "Node.js not found. Install from https://nodejs.org/"
command -v curl   >/dev/null 2>&1  || error "curl not found. Run: sudo apt-get install -y curl"
command -v git    >/dev/null 2>&1  || error "git not found. Run: sudo apt-get install -y git"

# Docker must be running
docker info >/dev/null 2>&1 || error "Docker daemon is not running. Start with: sudo systemctl start docker"

info "All prerequisites satisfied."

# ── 2. Download fabric-samples + binaries if needed ──────────────────────────
if [[ ! -d "${FABRIC_SAMPLES_DIR}" ]]; then
  info "Downloading Hyperledger Fabric samples and binaries (this may take a few minutes) …"
  mkdir -p "${FABRIC_SAMPLES_DIR}"
  cd "${HOME}"
  curl -sSL https://bit.ly/2ysbOFE | bash -s -- 2.5.15 1.5.15 -d -s
  info "fabric-samples downloaded to ${FABRIC_SAMPLES_DIR}"
else
  info "fabric-samples already present at ${FABRIC_SAMPLES_DIR}"
fi

# Ensure peer / orderer binaries are on PATH
export PATH="${FABRIC_SAMPLES_DIR}/bin:${PATH}"
export FABRIC_CFG_PATH="${FABRIC_SAMPLES_DIR}/config"

command -v peer >/dev/null 2>&1 || error "peer binary not found after download. Check ${FABRIC_SAMPLES_DIR}/bin"

# ── 3. Build the TypeScript chaincode ─────────────────────────────────────────
info "Building TypeScript chaincode …"
cd "${CHAINCODE_PATH}"
npm install --silent
npm run build
info "Chaincode built successfully."

# ── 4. Start the test network ─────────────────────────────────────────────────
cd "${TEST_NETWORK}"

info "Bringing down any previous network state …"
./network.sh down 2>/dev/null || true

info "Starting Fabric test network and creating channel '${CHANNEL}' …"
./network.sh up createChannel -c "${CHANNEL}" -ca

# ── 5. Deploy chaincode ───────────────────────────────────────────────────────
info "Deploying '${CC_NAME}' chaincode …"
./network.sh deployCC \
  -ccn "${CC_NAME}" \
  -ccp "${CHAINCODE_PATH}" \
  -ccl typescript \
  -c  "${CHANNEL}"

# ── 6. Set up environment for Org1 peer CLI ───────────────────────────────────
export CORE_PEER_TLS_ENABLED=true
export CORE_PEER_LOCALMSPID="Org1MSP"
export CORE_PEER_TLS_ROOTCERT_FILE="${TEST_NETWORK}/organizations/peerOrganizations/org1.example.com/peers/peer0.org1.example.com/tls/ca.crt"
export CORE_PEER_MSPCONFIGPATH="${TEST_NETWORK}/organizations/peerOrganizations/org1.example.com/users/Admin@org1.example.com/msp"
export CORE_PEER_ADDRESS=localhost:7051
export ORDERER_CA="${TEST_NETWORK}/organizations/ordererOrganizations/example.com/orderers/orderer.example.com/msp/tlscacerts/tlsca.example.com-cert.pem"

# ── 7. Initialise the ledger ──────────────────────────────────────────────────
info "Initialising ledger with 7 seed routes …"
peer chaincode invoke \
  -o localhost:7050 \
  --ordererTLSHostnameOverride orderer.example.com \
  --tls --cafile "${ORDERER_CA}" \
  -C "${CHANNEL}" -n "${CC_NAME}" \
  --peerAddresses localhost:7051 --tlsRootCertFiles "${CORE_PEER_TLS_ROOTCERT_FILE}" \
  --peerAddresses localhost:9051 --tlsRootCertFiles "${TEST_NETWORK}/organizations/peerOrganizations/org2.example.com/peers/peer0.org2.example.com/tls/ca.crt" \
  -c '{"function":"InitLedger","Args":[]}'

sleep 3   # wait for block commit

# ── 8. Quick smoke test ───────────────────────────────────────────────────────
info "Running smoke-test query (QueryAllRoutes) …"
peer chaincode query \
  -C "${CHANNEL}" -n "${CC_NAME}" \
  -c '{"function":"QueryAllRoutes","Args":[]}' \
  | python3 -m json.tool --no-ensure-ascii 2>/dev/null | head -40

echo ""
echo -e "${GREEN}════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}  Supply Chain chaincode deployed and ready!             ${NC}"
echo -e "${GREEN}  Channel  : ${CHANNEL}                                  ${NC}"
echo -e "${GREEN}  Chaincode: ${CC_NAME}                                  ${NC}"
echo -e "${GREEN}  See DEMO.md for full interaction commands.             ${NC}"
echo -e "${GREEN}════════════════════════════════════════════════════════${NC}"
