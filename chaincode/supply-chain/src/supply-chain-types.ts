/*
 * Supply Chain Analysis — Hyperledger Fabric Chaincode
 * Type definitions for on-chain assets
 */

/** A single HMB stop recorded within a route */
export interface HMBStop {
  hmbId: string;           // Unique HMB identifier (e.g. "HMB-101")
  name: string;            // Village / locality name
  lat: number;             // GPS latitude
  lon: number;             // GPS longitude
  expectedMilkQty: number; // Expected milk collection in litres
}

/**
 * A milk collection route from/to the Chilling Center.
 * Stored on-ledger; keyed by routeId.
 */
export interface CollectionRoute {
  docType: "collectionRoute";
  routeId: string;         // e.g. "ROUTE-01"
  plantId: string;         // Chilling Center plant code, e.g. "1142"
  plantName: string;       // Human-readable CC name, e.g. "Uthangarai CC"
  vehicleType: string;     // e.g. "Tanker" | "Minivan"
  maxCapacityLitres: number;
  currentLoadLitres: number;
  stops: HMBStop[];        // Ordered list of stops
  createdAt: string;       // ISO-8601 timestamp
  updatedAt: string;
}

/**
 * Records an HMB-to-route assignment decided by the optimizer.
 * Stored on-ledger; keyed by assignmentId.
 */
export interface HMBAssignment {
  docType: "hmbAssignment";
  assignmentId: string;    // e.g. "ASSIGN-20240501-001"
  hmbId: string;
  hmbName: string;
  lat: number;
  lon: number;
  expectedMilkQty: number;
  assignedRouteId: string;
  insertionPosition: number; // Index in stop list after assignment
  extraDistanceKm: number;
  totalRouteKmAfter: number;
  optimizerScore: number;    // Composite score (lower = better)
  assignedAt: string;        // ISO-8601 timestamp
  assignedBy: string;        // Operator / system identity
}
