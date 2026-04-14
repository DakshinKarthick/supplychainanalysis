#!/usr/bin/env bash
# =============================================================================
# demo-transcript.sh — Step-by-step demo of all chaincode operations
#
# Run this AFTER fabric/setup.sh has deployed the chaincode.
#
# Usage:
#   bash fabric/demo-transcript.sh
#
# =============================================================================

set -euo pipefail

FABRIC_SAMPLES_DIR="${HOME}/fabric-samples"
TEST_NETWORK="${FABRIC_SAMPLES_DIR}/test-network"
CHANNEL="mychannel"
CC_NAME="supply-chain"

export PATH="${FABRIC_SAMPLES_DIR}/bin:${PATH}"
export FABRIC_CFG_PATH="${FABRIC_SAMPLES_DIR}/config"
export CORE_PEER_TLS_ENABLED=true

# ── Colour helpers ────────────────────────────────────────────────────────────
BLUE='\033[0;34m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'
step()  { echo -e "\n${BLUE}━━━  $*  ━━━${NC}"; }
info()  { echo -e "${GREEN}▶${NC} $*"; }
result(){ echo -e "${YELLOW}$*${NC}"; }

# ── Org1 CLI environment ──────────────────────────────────────────────────────
org1_env() {
  export CORE_PEER_LOCALMSPID="Org1MSP"
  export CORE_PEER_TLS_ROOTCERT_FILE="${TEST_NETWORK}/organizations/peerOrganizations/org1.example.com/peers/peer0.org1.example.com/tls/ca.crt"
  export CORE_PEER_MSPCONFIGPATH="${TEST_NETWORK}/organizations/peerOrganizations/org1.example.com/users/Admin@org1.example.com/msp"
  export CORE_PEER_ADDRESS=localhost:7051
}

# ── Org2 CLI environment ──────────────────────────────────────────────────────
org2_env() {
  export CORE_PEER_LOCALMSPID="Org2MSP"
  export CORE_PEER_TLS_ROOTCERT_FILE="${TEST_NETWORK}/organizations/peerOrganizations/org2.example.com/peers/peer0.org2.example.com/tls/ca.crt"
  export CORE_PEER_MSPCONFIGPATH="${TEST_NETWORK}/organizations/peerOrganizations/org2.example.com/users/Admin@org2.example.com/msp"
  export CORE_PEER_ADDRESS=localhost:9051
}

ORDERER_CA="${TEST_NETWORK}/organizations/ordererOrganizations/example.com/orderers/orderer.example.com/msp/tlscacerts/tlsca.example.com-cert.pem"
ORG1_PEER_TLS="${TEST_NETWORK}/organizations/peerOrganizations/org1.example.com/peers/peer0.org1.example.com/tls/ca.crt"
ORG2_PEER_TLS="${TEST_NETWORK}/organizations/peerOrganizations/org2.example.com/peers/peer0.org2.example.com/tls/ca.crt"

# Helper: invoke on both peers (endorsement from Org1 + Org2)
invoke() {
  local fn_args="$1"
  org1_env
  peer chaincode invoke \
    -o localhost:7050 \
    --ordererTLSHostnameOverride orderer.example.com \
    --tls --cafile "${ORDERER_CA}" \
    -C "${CHANNEL}" -n "${CC_NAME}" \
    --peerAddresses localhost:7051 --tlsRootCertFiles "${ORG1_PEER_TLS}" \
    --peerAddresses localhost:9051 --tlsRootCertFiles "${ORG2_PEER_TLS}" \
    -c "${fn_args}"
  sleep 3
}

# Helper: query via Org1 peer
query() {
  local fn_args="$1"
  org1_env
  peer chaincode query \
    -C "${CHANNEL}" -n "${CC_NAME}" \
    -c "${fn_args}"
}

# Helper: pretty-print JSON
pp() { python3 -m json.tool --no-ensure-ascii 2>/dev/null || cat; }

# =============================================================================
echo ""
echo -e "${GREEN}╔══════════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║   Supply Chain Analysis — Chaincode Demo Transcript          ║${NC}"
echo -e "${GREEN}║   Channel: ${CHANNEL}   Chaincode: ${CC_NAME}               ║${NC}"
echo -e "${GREEN}╚══════════════════════════════════════════════════════════════╝${NC}"

# ── STEP 1: Query all seeded routes ──────────────────────────────────────────
step "STEP 1 — Query all existing collection routes"
info "Command: QueryAllRoutes"
echo ""
info "Running:"
echo "  peer chaincode query -C mychannel -n supply-chain -c '{\"function\":\"QueryAllRoutes\",\"Args\":[]}'"
echo ""
result "$(query '{"function":"QueryAllRoutes","Args":[]}' | pp | head -80)"

# ── STEP 2: Query a specific route ───────────────────────────────────────────
step "STEP 2 — Query a single route (ROUTE-04)"
info "Command: QueryRoute"
echo ""
info "Running:"
echo "  peer chaincode query -C mychannel -n supply-chain -c '{\"function\":\"QueryRoute\",\"Args\":[\"ROUTE-04\"]}'"
echo ""
result "$(query '{"function":"QueryRoute","Args":["ROUTE-04"]}' | pp)"

# ── STEP 3: Simulate optimizer output → record new HMB assignment ─────────────
step "STEP 3 — Record an HMB assignment (optimizer output → blockchain)"
info "Scenario: The Python optimizer recommends assigning a new HMB"
info "  'Thoppur Village' (lat 12.35, lon 78.55, 100 L) to ROUTE-04"
info "  Insertion at position 3, adding 2.4 km, score 0.312"
echo ""
ASSIGN1_ARGS='{"function":"RecordHMBAssignment","Args":["ASSIGN-001","HMB-NEW-01","Thoppur Village","12.35","78.55","100","ROUTE-04","3","2.4","95.0","0.312","SupplyChainOptimizer-v2"]}'

info "Running:"
echo "  peer chaincode invoke ... -c '${ASSIGN1_ARGS}'"
echo ""
invoke "${ASSIGN1_ARGS}"
info "Transaction committed."

# ── STEP 4: Verify assignment was persisted ────────────────────────────────────
step "STEP 4 — Verify assignment record on the ledger"
info "Command: QueryAssignment ASSIGN-001"
echo ""
result "$(query '{"function":"QueryAssignment","Args":["ASSIGN-001"]}' | pp)"

# ── STEP 5: Verify route was updated (load + new stop) ────────────────────────
step "STEP 5 — Verify ROUTE-04 now includes the new HMB stop"
info "ROUTE-04 currentLoadLitres should have increased by 100 L"
echo ""
result "$(query '{"function":"QueryRoute","Args":["ROUTE-04"]}' | pp)"

# ── STEP 6: Add another assignment to a different route ───────────────────────
step "STEP 6 — Assign a second new HMB to ROUTE-02"
info "Scenario: 'Pannandur' (lat 12.17, lon 78.21, 80 L) → ROUTE-02"
echo ""
invoke '{"function":"RecordHMBAssignment","Args":["ASSIGN-002","HMB-NEW-02","Pannandur","12.17","78.21","80","ROUTE-02","2","1.8","76.5","0.287","SupplyChainOptimizer-v2"]}'
info "Transaction committed."

result "$(query '{"function":"QueryAssignment","Args":["ASSIGN-002"]}' | pp)"

# ── STEP 7: Query full history of ROUTE-04 ────────────────────────────────────
step "STEP 7 — View transaction history for ROUTE-04 on the blockchain"
info "Command: GetAssignmentHistory ROUTE-04"
info "(Shows every ledger version: initial seed + after assignment)"
echo ""
result "$(query '{"function":"GetAssignmentHistory","Args":["ROUTE-04"]}' | pp)"

# ── STEP 8: Create a brand-new route ─────────────────────────────────────────
step "STEP 8 — Create a new route (admin: adding Plant-1142 Route-08)"
info "Command: CreateRoute"
echo ""
invoke '{"function":"CreateRoute","Args":["ROUTE-08","1142","Uthangarai CC","Tanker","2000"]}'
info "ROUTE-08 created."

result "$(query '{"function":"QueryRoute","Args":["ROUTE-08"]}' | pp)"

# ── STEP 9: Update a route's load ────────────────────────────────────────────
step "STEP 9 — Update load on ROUTE-08 (manual correction)"
info "Command: UpdateRouteLoad ROUTE-08 → 500 L"
echo ""
invoke '{"function":"UpdateRouteLoad","Args":["ROUTE-08","500"]}'
info "Load updated."

result "$(query '{"function":"QueryRoute","Args":["ROUTE-08"]}' | pp)"

# ── STEP 10: Query all routes (show complete state) ───────────────────────────
step "STEP 10 — Final state: QueryAllRoutes"
info "Shows all 8 routes with their latest load and stops"
echo ""
result "$(query '{"function":"QueryAllRoutes","Args":[]}' | pp)"

# ── STEP 11: Org2 cross-check ────────────────────────────────────────────────
step "STEP 11 — Cross-organisation verification (Org2 queries the ledger)"
info "Switching to Org2 peer …"
org2_env
info "Running QueryRoute ROUTE-04 from Org2:"
echo ""
result "$(peer chaincode query -C "${CHANNEL}" -n "${CC_NAME}" -c '{"function":"QueryRoute","Args":["ROUTE-04"]}' | pp)"

echo ""
echo -e "${GREEN}╔══════════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║   Demo complete!  All transactions verified on both orgs.    ║${NC}"
echo -e "${GREEN}╚══════════════════════════════════════════════════════════════╝${NC}"
