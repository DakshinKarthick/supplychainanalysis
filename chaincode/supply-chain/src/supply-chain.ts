/*
 * Supply Chain Analysis — Hyperledger Fabric Chaincode
 * Core smart-contract logic
 *
 * Fabric SDK: fabric-contract-api (TypeScript)
 */

import { Context, Contract, Info, Returns, Transaction } from "fabric-contract-api";
import { CollectionRoute, HMBAssignment, HMBStop } from "./supply-chain-types";

@Info({
  title: "SupplyChainContract",
  description: "Smart contract for HMB milk-route management on Hyperledger Fabric",
})
export class SupplyChainContract extends Contract {
  // ────────────────────────────────────────────────────────────────────────────
  // Ledger Initialisation
  // ────────────────────────────────────────────────────────────────────────────

  /**
   * InitLedger — seeds the ledger with the seven existing collection routes
   * from Uthangarai Chilling Center (Plant 1142).
   *
   * Call once after first deploy:
   *   peer chaincode invoke ... -c '{"function":"InitLedger","Args":[]}'
   */
  @Transaction()
  public async InitLedger(ctx: Context): Promise<void> {
    const now = new Date().toISOString();

    const routes: CollectionRoute[] = [
      {
        docType: "collectionRoute",
        routeId: "ROUTE-01",
        plantId: "1142",
        plantName: "Uthangarai CC",
        vehicleType: "Tanker",
        maxCapacityLitres: 2000,
        currentLoadLitres: 1450,
        stops: [
          { hmbId: "HMB-101", name: "Uthangarai", lat: 12.308573, lon: 78.535901, expectedMilkQty: 200 },
          { hmbId: "HMB-102", name: "Chinnakaruppur", lat: 12.33, lon: 78.51, expectedMilkQty: 250 },
          { hmbId: "HMB-103", name: "Kottaiyur", lat: 12.36, lon: 78.49, expectedMilkQty: 300 },
        ],
        createdAt: now,
        updatedAt: now,
      },
      {
        docType: "collectionRoute",
        routeId: "ROUTE-02",
        plantId: "1142",
        plantName: "Uthangarai CC",
        vehicleType: "Tanker",
        maxCapacityLitres: 2000,
        currentLoadLitres: 1100,
        stops: [
          { hmbId: "HMB-201", name: "Pennagaram", lat: 12.12, lon: 78.16, expectedMilkQty: 180 },
          { hmbId: "HMB-202", name: "Hanumanthapuram", lat: 12.15, lon: 78.19, expectedMilkQty: 220 },
          { hmbId: "HMB-203", name: "Kariyakovil", lat: 12.18, lon: 78.22, expectedMilkQty: 250 },
          { hmbId: "HMB-204", name: "Karuppur", lat: 12.21, lon: 78.25, expectedMilkQty: 200 },
          { hmbId: "HMB-205", name: "Olaipadi", lat: 12.24, lon: 78.28, expectedMilkQty: 250 },
        ],
        createdAt: now,
        updatedAt: now,
      },
      {
        docType: "collectionRoute",
        routeId: "ROUTE-03",
        plantId: "1142",
        plantName: "Uthangarai CC",
        vehicleType: "Minivan",
        maxCapacityLitres: 1200,
        currentLoadLitres: 980,
        stops: [
          { hmbId: "HMB-301", name: "Harur", lat: 12.05, lon: 78.48, expectedMilkQty: 150 },
          { hmbId: "HMB-302", name: "Kovilur", lat: 12.08, lon: 78.45, expectedMilkQty: 180 },
          { hmbId: "HMB-303", name: "Maniyambadi", lat: 12.11, lon: 78.42, expectedMilkQty: 200 },
          { hmbId: "HMB-304", name: "Solaikottai", lat: 12.14, lon: 78.39, expectedMilkQty: 200 },
          { hmbId: "HMB-305", name: "Thumbai", lat: 12.17, lon: 78.36, expectedMilkQty: 250 },
        ],
        createdAt: now,
        updatedAt: now,
      },
      {
        docType: "collectionRoute",
        routeId: "ROUTE-04",
        plantId: "1142",
        plantName: "Uthangarai CC",
        vehicleType: "Tanker",
        maxCapacityLitres: 2000,
        currentLoadLitres: 1300,
        stops: [
          { hmbId: "HMB-401", name: "Morappur", lat: 12.26, lon: 78.42, expectedMilkQty: 220 },
          { hmbId: "HMB-402", name: "Odaikadu", lat: 12.29, lon: 78.45, expectedMilkQty: 260 },
          { hmbId: "HMB-403", name: "Periyakurumbadi", lat: 12.32, lon: 78.48, expectedMilkQty: 280 },
          { hmbId: "HMB-404", name: "Kuppandapadi", lat: 12.35, lon: 78.51, expectedMilkQty: 240 },
          { hmbId: "HMB-405", name: "Vellariveli", lat: 12.38, lon: 78.54, expectedMilkQty: 300 },
        ],
        createdAt: now,
        updatedAt: now,
      },
      {
        docType: "collectionRoute",
        routeId: "ROUTE-05",
        plantId: "1142",
        plantName: "Uthangarai CC",
        vehicleType: "Minivan",
        maxCapacityLitres: 1200,
        currentLoadLitres: 750,
        stops: [
          { hmbId: "HMB-501", name: "Navalur Kuttappattu", lat: 12.42, lon: 78.38, expectedMilkQty: 120 },
          { hmbId: "HMB-502", name: "Gundalapadi", lat: 12.45, lon: 78.35, expectedMilkQty: 150 },
          { hmbId: "HMB-503", name: "Semmipalayam", lat: 12.48, lon: 78.32, expectedMilkQty: 180 },
          { hmbId: "HMB-504", name: "Chinnagoundanur", lat: 12.51, lon: 78.29, expectedMilkQty: 150 },
          { hmbId: "HMB-505", name: "Periyagoundanur", lat: 12.54, lon: 78.26, expectedMilkQty: 150 },
        ],
        createdAt: now,
        updatedAt: now,
      },
      {
        docType: "collectionRoute",
        routeId: "ROUTE-06",
        plantId: "1142",
        plantName: "Uthangarai CC",
        vehicleType: "Tanker",
        maxCapacityLitres: 2000,
        currentLoadLitres: 1650,
        stops: [
          { hmbId: "HMB-601", name: "Krishnagiri", lat: 12.52, lon: 78.21, expectedMilkQty: 300 },
          { hmbId: "HMB-602", name: "Hosur Road", lat: 12.49, lon: 78.18, expectedMilkQty: 280 },
          { hmbId: "HMB-603", name: "Anchetti", lat: 12.46, lon: 78.15, expectedMilkQty: 250 },
          { hmbId: "HMB-604", name: "Bargur", lat: 12.43, lon: 78.12, expectedMilkQty: 320 },
          { hmbId: "HMB-605", name: "Thally", lat: 12.4, lon: 78.09, expectedMilkQty: 300 },
        ],
        createdAt: now,
        updatedAt: now,
      },
      {
        docType: "collectionRoute",
        routeId: "ROUTE-07",
        plantId: "1142",
        plantName: "Uthangarai CC",
        vehicleType: "Tanker",
        maxCapacityLitres: 2000,
        currentLoadLitres: 1200,
        stops: [
          { hmbId: "HMB-701", name: "Palacode", lat: 12.22, lon: 78.35, expectedMilkQty: 200 },
          { hmbId: "HMB-702", name: "Kadambur", lat: 12.25, lon: 78.32, expectedMilkQty: 220 },
          { hmbId: "HMB-703", name: "Nallampalli", lat: 12.28, lon: 78.29, expectedMilkQty: 250 },
          { hmbId: "HMB-704", name: "Dharmapuri", lat: 12.31, lon: 78.26, expectedMilkQty: 260 },
          { hmbId: "HMB-705", name: "Marandahalli", lat: 12.34, lon: 78.23, expectedMilkQty: 270 },
        ],
        createdAt: now,
        updatedAt: now,
      },
    ];

    for (const route of routes) {
      await ctx.stub.putState(route.routeId, Buffer.from(JSON.stringify(route)));
    }

    console.log("Ledger initialised with 7 collection routes for Plant 1142 (Uthangarai CC)");
  }

  // ────────────────────────────────────────────────────────────────────────────
  // Route CRUD
  // ────────────────────────────────────────────────────────────────────────────

  /**
   * CreateRoute — add a new route (admin function).
   *
   * Example:
   *   peer chaincode invoke ... -c '{"function":"CreateRoute","Args":[
   *     "ROUTE-08","1142","Uthangarai CC","Tanker","2000"
   *   ]}'
   */
  @Transaction()
  public async CreateRoute(
    ctx: Context,
    routeId: string,
    plantId: string,
    plantName: string,
    vehicleType: string,
    maxCapacityLitres: string,
  ): Promise<void> {
    const exists = await this._routeExists(ctx, routeId);
    if (exists) {
      throw new Error(`Route ${routeId} already exists`);
    }

    const now = new Date().toISOString();
    const route: CollectionRoute = {
      docType: "collectionRoute",
      routeId,
      plantId,
      plantName,
      vehicleType,
      maxCapacityLitres: parseFloat(maxCapacityLitres),
      currentLoadLitres: 0,
      stops: [],
      createdAt: now,
      updatedAt: now,
    };

    await ctx.stub.putState(routeId, Buffer.from(JSON.stringify(route)));
  }

  /**
   * QueryRoute — retrieve a single route.
   *
   * Example:
   *   peer chaincode query ... -c '{"function":"QueryRoute","Args":["ROUTE-01"]}'
   */
  @Transaction(false)
  @Returns("string")
  public async QueryRoute(ctx: Context, routeId: string): Promise<string> {
    const data = await ctx.stub.getState(routeId);
    if (!data || data.length === 0) {
      throw new Error(`Route ${routeId} not found`);
    }
    return data.toString();
  }

  /**
   * QueryAllRoutes — list all collection routes.
   *
   * Example:
   *   peer chaincode query ... -c '{"function":"QueryAllRoutes","Args":[]}'
   */
  @Transaction(false)
  @Returns("string")
  public async QueryAllRoutes(ctx: Context): Promise<string> {
    const iterator = await ctx.stub.getStateByRange("ROUTE-", "ROUTE-\uFFFF");
    const results: CollectionRoute[] = [];

    let result = await iterator.next();
    while (!result.done) {
      const value = result.value.value;
      if (value && value.length > 0) {
        results.push(JSON.parse(value.toString()) as CollectionRoute);
      }
      result = await iterator.next();
    }

    await iterator.close();
    return JSON.stringify(results);
  }

  // ────────────────────────────────────────────────────────────────────────────
  // HMB Assignment
  // ────────────────────────────────────────────────────────────────────────────

  /**
   * RecordHMBAssignment — write an optimizer-generated assignment to the ledger.
   * Called by the Python optimizer after it has computed the best route.
   *
   * Example:
   *   peer chaincode invoke ... -c '{"function":"RecordHMBAssignment","Args":[
   *     "ASSIGN-001","HMB-NEW","Thoppur Village",
   *     "12.35","78.55","100",
   *     "ROUTE-04","3","2.4","92.6","0.312",
   *     "SupplyChainOptimizer-v2"
   *   ]}'
   */
  @Transaction()
  public async RecordHMBAssignment(
    ctx: Context,
    assignmentId: string,
    hmbId: string,
    hmbName: string,
    lat: string,
    lon: string,
    expectedMilkQty: string,
    assignedRouteId: string,
    insertionPosition: string,
    extraDistanceKm: string,
    totalRouteKmAfter: string,
    optimizerScore: string,
    assignedBy: string,
  ): Promise<void> {
    // Verify the target route exists
    const routeData = await ctx.stub.getState(assignedRouteId);
    if (!routeData || routeData.length === 0) {
      throw new Error(`Cannot assign to non-existent route ${assignedRouteId}`);
    }

    const milkQty = parseFloat(expectedMilkQty);
    const route: CollectionRoute = JSON.parse(routeData.toString());

    // Capacity check (also enforced by Python optimizer, but double-checked on-chain)
    if (route.currentLoadLitres + milkQty > route.maxCapacityLitres) {
      throw new Error(
        `Route ${assignedRouteId} would exceed capacity: ` +
          `current=${route.currentLoadLitres}L, adding=${milkQty}L, max=${route.maxCapacityLitres}L`,
      );
    }

    const now = new Date().toISOString();

    // Persist assignment record
    const assignment: HMBAssignment = {
      docType: "hmbAssignment",
      assignmentId,
      hmbId,
      hmbName,
      lat: parseFloat(lat),
      lon: parseFloat(lon),
      expectedMilkQty: milkQty,
      assignedRouteId,
      insertionPosition: parseInt(insertionPosition, 10),
      extraDistanceKm: parseFloat(extraDistanceKm),
      totalRouteKmAfter: parseFloat(totalRouteKmAfter),
      optimizerScore: parseFloat(optimizerScore),
      assignedAt: now,
      assignedBy,
    };

    await ctx.stub.putState(assignmentId, Buffer.from(JSON.stringify(assignment)));

    // Update route: add stop and update load
    const newStop: HMBStop = {
      hmbId,
      name: hmbName,
      lat: parseFloat(lat),
      lon: parseFloat(lon),
      expectedMilkQty: milkQty,
    };

    const pos = parseInt(insertionPosition, 10);
    route.stops.splice(pos, 0, newStop);
    route.currentLoadLitres += milkQty;
    route.updatedAt = now;

    await ctx.stub.putState(assignedRouteId, Buffer.from(JSON.stringify(route)));
  }

  /**
   * QueryAssignment — retrieve a specific assignment record.
   *
   * Example:
   *   peer chaincode query ... -c '{"function":"QueryAssignment","Args":["ASSIGN-001"]}'
   */
  @Transaction(false)
  @Returns("string")
  public async QueryAssignment(ctx: Context, assignmentId: string): Promise<string> {
    const data = await ctx.stub.getState(assignmentId);
    if (!data || data.length === 0) {
      throw new Error(`Assignment ${assignmentId} not found`);
    }
    return data.toString();
  }

  /**
   * GetAssignmentHistory — retrieve the full history (all versions) of a key.
   * Requires the peer to be started with --peer.ledger.state.couchdbConfig
   * or history DB enabled.
   *
   * Example:
   *   peer chaincode query ... -c '{"function":"GetAssignmentHistory","Args":["ROUTE-04"]}'
   */
  @Transaction(false)
  @Returns("string")
  public async GetAssignmentHistory(ctx: Context, key: string): Promise<string> {
    const iterator = await ctx.stub.getHistoryForKey(key);
    const history: object[] = [];

    let result = await iterator.next();
    while (!result.done) {
      const record: Record<string, unknown> = {
        txId: result.value.txId,
        timestamp: result.value.timestamp,
        isDelete: result.value.isDelete,
      };
      if (!result.value.isDelete && result.value.value) {
        record["data"] = JSON.parse(result.value.value.toString());
      }
      history.push(record);
      result = await iterator.next();
    }

    await iterator.close();
    return JSON.stringify(history);
  }

  /**
   * UpdateRouteLoad — manually correct the current load on a route
   * (e.g. after a seasonal adjustment or data correction).
   *
   * Example:
   *   peer chaincode invoke ... -c '{"function":"UpdateRouteLoad","Args":["ROUTE-04","1400"]}'
   */
  @Transaction()
  public async UpdateRouteLoad(
    ctx: Context,
    routeId: string,
    newLoadLitres: string,
  ): Promise<void> {
    const data = await ctx.stub.getState(routeId);
    if (!data || data.length === 0) {
      throw new Error(`Route ${routeId} not found`);
    }

    const route: CollectionRoute = JSON.parse(data.toString());
    const newLoad = parseFloat(newLoadLitres);

    if (newLoad < 0 || newLoad > route.maxCapacityLitres) {
      throw new Error(
        `Load ${newLoad}L is out of valid range [0, ${route.maxCapacityLitres}]`,
      );
    }

    route.currentLoadLitres = newLoad;
    route.updatedAt = new Date().toISOString();

    await ctx.stub.putState(routeId, Buffer.from(JSON.stringify(route)));
  }

  // ────────────────────────────────────────────────────────────────────────────
  // Helpers
  // ────────────────────────────────────────────────────────────────────────────

  private async _routeExists(ctx: Context, routeId: string): Promise<boolean> {
    const data = await ctx.stub.getState(routeId);
    return data !== undefined && data.length > 0;
  }
}
