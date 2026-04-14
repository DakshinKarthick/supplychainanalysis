/*
 * Supply Chain Analysis — Hyperledger Fabric Chaincode
 * Entry point — re-exports the contract class for the fabric-contract-api loader
 */

import { SupplyChainContract } from "./src/supply-chain";

export { SupplyChainContract };
export const contracts: typeof SupplyChainContract[] = [SupplyChainContract];
