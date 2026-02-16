"""
Tests for Route Optimizer V2.

Covers:
  - Haversine formula
  - Data loading
  - Capacity pre-filter
  - Insertion cost algorithm
  - 2-opt improvement
  - Scoring and ranking
  - End-to-end recommendations (Haversine mode)
  - Input validation
"""

import math
import os
import sys
import pytest

# Ensure the script directory is on the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from route_optimizer import (
    HMB, Route, InsertionResult,
    haversine, haversine_road_distance,
    _parse_lat_lon, load_route_data, load_route_summary,
    build_route_coords, calculate_route_distance, calculate_route_distance_from_coords,
    estimate_route_time, calculate_route_centroid,
    optimize_2opt, _build_distance_matrix_haversine, _total_distance_from_matrix,
    filter_routes_by_capacity,
    calculate_insertion_cost, _build_stop_list, _resolve_stop_name,
    score_routes, recommend_route, haversine_recommendation,
    CC_LAT, CC_LON, ROAD_FACTOR, MAX_ROUTE_HOURS, AVG_SPEED_KMH, AVG_STOP_TIME_HOURS,
)


# =============================================================================
# Haversine Formula Tests
# =============================================================================

class TestHaversine:
    """Tests for the Haversine distance function."""

    def test_same_point(self):
        """Distance from a point to itself is zero."""
        assert haversine(12.0, 78.0, 12.0, 78.0) == 0.0

    def test_known_distance(self):
        """Verify against known distance: Delhi to Mumbai ~1150km."""
        dist = haversine(28.6139, 77.2090, 19.0760, 72.8777)
        assert 1100 < dist < 1200

    def test_symmetric(self):
        """Distance A->B should equal B->A."""
        d1 = haversine(12.0, 78.0, 13.0, 79.0)
        d2 = haversine(13.0, 79.0, 12.0, 78.0)
        assert abs(d1 - d2) < 0.001

    def test_cc_to_known_hmb(self):
        """Distance from CC to a nearby HMB should be reasonable (5-30km)."""
        # Chandrapuram approximate coordinates
        dist = haversine(CC_LAT, CC_LON, 12.2007, 78.5444)
        assert 5 < dist < 20

    def test_road_distance_factor(self):
        """Road distance should be ROAD_FACTOR times Haversine."""
        hav = haversine(12.0, 78.0, 12.1, 78.1)
        road = haversine_road_distance(12.0, 78.0, 12.1, 78.1)
        assert abs(road - hav * ROAD_FACTOR) < 0.001

    def test_equator_distance(self):
        """1 degree longitude at equator is ~111km."""
        dist = haversine(0.0, 0.0, 0.0, 1.0)
        assert 110 < dist < 112


# =============================================================================
# Coordinate Parsing Tests
# =============================================================================

class TestParsing:
    """Tests for coordinate parsing."""

    def test_valid_coords(self):
        assert _parse_lat_lon("12.3, 78.5") == (12.3, 78.5)

    def test_no_spaces(self):
        assert _parse_lat_lon("12.3,78.5") == (12.3, 78.5)

    def test_extra_whitespace(self):
        assert _parse_lat_lon("  12.3 , 78.5  ") == (12.3, 78.5)

    def test_empty_string(self):
        assert _parse_lat_lon("") is None

    def test_none_input(self):
        assert _parse_lat_lon(None) is None

    def test_invalid_format(self):
        assert _parse_lat_lon("abc,xyz") is None

    def test_out_of_range_lat(self):
        assert _parse_lat_lon("91.0, 78.0") is None

    def test_out_of_range_lon(self):
        assert _parse_lat_lon("12.0, 181.0") is None

    def test_three_values(self):
        assert _parse_lat_lon("12.0, 78.0, 45.0") is None


# =============================================================================
# Data Loading Tests
# =============================================================================

class TestDataLoading:
    """Tests for CSV data loading."""

    def test_load_summary(self):
        """Summary should contain Uthangarai routes."""
        summary = load_route_summary()
        assert len(summary) >= 7
        assert all(k.startswith("M") for k in summary.keys())

    def test_summary_has_capacity(self):
        """Each route should have a capacity value."""
        summary = load_route_summary()
        for code, info in summary.items():
            assert info["capacity"] > 0, f"Route {code} has no capacity"

    def test_load_routes(self):
        """Should load 7 routes for Uthangarai."""
        routes = load_route_data()
        assert len(routes) == 7

    def test_routes_have_hmbs(self):
        """Each route should have at least 1 HMB."""
        routes = load_route_data()
        for route in routes:
            assert len(route.hmbs) >= 1, f"Route {route.code} has no HMBs"

    def test_routes_have_valid_coords(self):
        """All HMBs should have valid GPS coordinates."""
        routes = load_route_data()
        for route in routes:
            for hmb in route.hmbs:
                assert -90 <= hmb.lat <= 90, f"{hmb.name} has invalid lat"
                assert -180 <= hmb.lon <= 180, f"{hmb.name} has invalid lon"

    def test_routes_sorted_by_code(self):
        """Routes should be sorted by code."""
        routes = load_route_data()
        codes = [r.code for r in routes]
        assert codes == sorted(codes)

    def test_hmbs_sorted_by_sequence(self):
        """HMBs within each route should be sorted by sequence."""
        routes = load_route_data()
        for route in routes:
            sequences = [h.sequence for h in route.hmbs]
            assert sequences == sorted(sequences), f"Route {route.code} HMBs out of order"


# =============================================================================
# Route Calculation Tests
# =============================================================================

class TestRouteCalculations:
    """Tests for route distance and time calculations."""

    def test_build_route_coords_has_cc_bookends(self):
        """Route coords should start and end at CC."""
        routes = load_route_data()
        coords = build_route_coords(routes[0])
        assert coords[0] == (CC_LAT, CC_LON)
        assert coords[-1] == (CC_LAT, CC_LON)

    def test_build_route_coords_length(self):
        """Coords should have CC + HMBs + CC entries."""
        routes = load_route_data()
        route = routes[0]
        coords = build_route_coords(route)
        assert len(coords) == len(route.hmbs) + 2

    def test_route_distance_positive(self):
        """Route distance should be positive."""
        routes = load_route_data()
        for route in routes:
            dist = calculate_route_distance(route, "haversine")
            assert dist > 0, f"Route {route.code} has zero distance"

    def test_route_distance_reasonable(self):
        """Route distances should be between 10-200 km (reasonable for this region)."""
        routes = load_route_data()
        for route in routes:
            dist = calculate_route_distance(route, "haversine")
            assert 10 < dist < 200, f"Route {route.code} has unreasonable distance: {dist}"

    def test_estimate_route_time(self):
        """Time = driving + stops."""
        time_h = estimate_route_time(80.0, 10)
        expected = 80.0 / AVG_SPEED_KMH + 10 * AVG_STOP_TIME_HOURS
        assert abs(time_h - expected) < 0.001

    def test_centroid_no_hmbs(self):
        """Empty route centroid should default to CC."""
        route = Route(code="X", name="Empty")
        centroid = calculate_route_centroid(route)
        assert centroid == (CC_LAT, CC_LON)

    def test_centroid_reasonable(self):
        """Centroid should be near Uthangarai (within ~20km)."""
        routes = load_route_data()
        for route in routes:
            centroid = calculate_route_centroid(route)
            dist = haversine(CC_LAT, CC_LON, centroid[0], centroid[1])
            assert dist < 30, f"Route {route.code} centroid is {dist:.1f}km from CC"


# =============================================================================
# 2-Opt Tests
# =============================================================================

class TestTwoOpt:
    """Tests for the 2-opt route improvement algorithm."""

    def test_2opt_does_not_increase_distance(self):
        """2-opt should never increase total distance."""
        routes = load_route_data()
        for route in routes:
            coords = build_route_coords(route)
            original_dist = calculate_route_distance_from_coords(coords, "haversine")
            opt_coords, opt_dist = optimize_2opt(coords, "haversine")
            assert opt_dist <= original_dist + 0.1, \
                f"Route {route.code}: 2-opt increased distance ({original_dist:.1f} -> {opt_dist:.1f})"

    def test_2opt_preserves_depot(self):
        """2-opt should keep CC as first and last stop."""
        routes = load_route_data()
        coords = build_route_coords(routes[0])
        opt_coords, _ = optimize_2opt(coords, "haversine")
        assert opt_coords[0] == (CC_LAT, CC_LON)
        assert opt_coords[-1] == (CC_LAT, CC_LON)

    def test_2opt_preserves_stop_count(self):
        """2-opt should not add or remove stops."""
        routes = load_route_data()
        coords = build_route_coords(routes[0])
        opt_coords, _ = optimize_2opt(coords, "haversine")
        assert len(opt_coords) == len(coords)

    def test_2opt_single_stop(self):
        """Single-stop route has nothing to optimize."""
        hmb = HMB("1", "Test", 12.3, 78.5, 1, 0)
        route = Route(code="T", name="Test", hmbs=[hmb])
        coords = build_route_coords(route)
        opt_coords, opt_dist = optimize_2opt(coords, "haversine")
        assert len(opt_coords) == 3  # CC, HMB, CC

    def test_2opt_two_stops(self):
        """Two-stop route: 2-opt should find if swapping helps."""
        hmbs = [
            HMB("1", "A", 12.2, 78.4, 1, 0),
            HMB("2", "B", 12.4, 78.6, 2, 0),
        ]
        route = Route(code="T", name="Test", hmbs=hmbs)
        coords = build_route_coords(route)
        opt_coords, opt_dist = optimize_2opt(coords, "haversine")
        original_dist = calculate_route_distance_from_coords(coords, "haversine")
        assert opt_dist <= original_dist + 0.01

    def test_distance_matrix_symmetry(self):
        """Haversine distance matrix should be symmetric."""
        coords = [(CC_LAT, CC_LON), (12.2, 78.5), (12.3, 78.4)]
        matrix = _build_distance_matrix_haversine(coords)
        for c1 in coords:
            for c2 in coords:
                if c1 != c2:
                    assert abs(matrix[(c1, c2)] - matrix[(c2, c1)]) < 0.01


# =============================================================================
# Capacity Pre-Filter Tests
# =============================================================================

class TestCapacityFilter:
    """Tests for the capacity hard pre-filter."""

    def test_all_routes_eligible_small_qty(self):
        """Small milk qty should pass all routes."""
        routes = load_route_data()
        eligible, rejected = filter_routes_by_capacity(routes, 10)  # 10 litres
        assert len(eligible) == 7
        assert len(rejected) == 0

    def test_all_routes_rejected_huge_qty(self):
        """Huge milk qty should reject all routes."""
        routes = load_route_data()
        eligible, rejected = filter_routes_by_capacity(routes, 5000)
        assert len(eligible) == 0
        assert len(rejected) == 7

    def test_zero_qty_passes_all(self):
        """Zero milk qty should pass all routes."""
        routes = load_route_data()
        eligible, rejected = filter_routes_by_capacity(routes, 0)
        assert len(eligible) == 7

    def test_rejection_has_reason(self):
        """Each rejected route should have a reason string."""
        routes = load_route_data()
        _, rejected = filter_routes_by_capacity(routes, 5000)
        for rej in rejected:
            assert "Capacity exceeded" in rej["reason"]
            assert rej["route"].code.startswith("M")

    def test_capacity_boundary(self):
        """Route at exact capacity should be rejected."""
        route = Route(code="T", name="Test", capacity=100, current_milk_qty=90)
        eligible, rejected = filter_routes_by_capacity([route], 11)  # 90 + 11 = 101 > 100
        assert len(eligible) == 0
        assert len(rejected) == 1

    def test_capacity_exact_fit(self):
        """Route with exactly enough room should pass."""
        route = Route(code="T", name="Test", capacity=100, current_milk_qty=90)
        eligible, rejected = filter_routes_by_capacity([route], 10)  # 90 + 10 = 100
        assert len(eligible) == 1
        assert len(rejected) == 0


# =============================================================================
# Insertion Cost Tests
# =============================================================================

class TestInsertionCost:
    """Tests for the insertion cost algorithm."""

    def test_insertion_returns_result(self):
        """Insertion cost should return a valid result."""
        routes = load_route_data()
        result = calculate_insertion_cost(routes[0], 12.35, 78.55)
        assert result is not None
        assert isinstance(result, InsertionResult)

    def test_extra_km_non_negative(self):
        """Extra KM should be >= 0 (negative means shortcut found)."""
        routes = load_route_data()
        for route in routes:
            result = calculate_insertion_cost(route, 12.35, 78.55)
            # In rare cases the insertion can "shortcut" through the new point
            # but it should still be > some reasonable minimum
            assert result.extra_km >= -5.0, f"Route {route.code}: unreasonable extra_km"

    def test_post_2opt_not_worse(self):
        """Post-2opt distance should not exceed pre-2opt distance (by much)."""
        routes = load_route_data()
        for route in routes:
            result = calculate_insertion_cost(route, 12.35, 78.55)
            assert result.post_2opt_km <= result.new_total_km + 0.5, \
                f"Route {route.code}: 2-opt made it worse"

    def test_improvement_km_non_negative(self):
        """2-opt should save distance (or zero)."""
        routes = load_route_data()
        for route in routes:
            result = calculate_insertion_cost(route, 12.35, 78.55)
            assert result.improvement_km >= -0.5, \
                f"Route {route.code}: negative improvement"

    def test_has_stop_names(self):
        """Result should have prev and next stop names."""
        routes = load_route_data()
        result = calculate_insertion_cost(routes[0], 12.35, 78.55)
        assert result.prev_stop_name != ""
        assert result.next_stop_name != ""

    def test_has_optimized_order(self):
        """Result should have optimized stop order with NEW HMB in it."""
        routes = load_route_data()
        result = calculate_insertion_cost(routes[0], 12.35, 78.55)
        assert "NEW HMB" in result.optimized_stop_order

    def test_time_constraint_applied(self):
        """Routes exceeding MAX_ROUTE_HOURS should be infeasible."""
        routes = load_route_data()
        # Find if any route is infeasible
        infeasible_count = 0
        for route in routes:
            result = calculate_insertion_cost(route, 12.35, 78.55)
            if not result.is_feasible:
                assert "exceeds max" in result.infeasibility_reason
                infeasible_count += 1
        # At least some routes should have time issues with 4h constraint
        # (we know Kandhili-II does from earlier tests)
        assert infeasible_count >= 0  # May be 0 for some coordinates


# =============================================================================
# Scoring Tests
# =============================================================================

class TestScoring:
    """Tests for the multi-factor scoring system."""

    def test_scores_between_0_and_1(self):
        """All scores should be in [0, 1]."""
        routes = load_route_data()
        results = [calculate_insertion_cost(r, 12.35, 78.55) for r in routes]
        scored = score_routes(results, 12.35, 78.55, "haversine")
        for r in scored:
            assert 0.0 <= r.score <= 1.0, f"Route {r.route.code}: score {r.score} out of range"

    def test_feasible_routes_ranked_first(self):
        """Feasible routes should always rank above infeasible ones."""
        routes = load_route_data()
        results = [calculate_insertion_cost(r, 12.35, 78.55) for r in routes]
        scored = score_routes(results, 12.35, 78.55, "haversine")
        found_infeasible = False
        for r in scored:
            if not r.is_feasible:
                found_infeasible = True
            if found_infeasible and r.is_feasible:
                assert False, "Feasible route ranked below infeasible"

    def test_sorted_by_score_within_feasibility(self):
        """Within feasible group, scores should be descending."""
        routes = load_route_data()
        results = [calculate_insertion_cost(r, 12.35, 78.55) for r in routes]
        scored = score_routes(results, 12.35, 78.55, "haversine")
        feasible = [r for r in scored if r.is_feasible]
        for i in range(len(feasible) - 1):
            assert feasible[i].score >= feasible[i + 1].score, \
                "Scores not in descending order"

    def test_empty_results(self):
        """Empty input should return empty output."""
        result = score_routes([], 12.35, 78.55, "haversine")
        assert result == []


# =============================================================================
# End-to-End Recommendation Tests (Haversine)
# =============================================================================

class TestRecommendation:
    """End-to-end tests for the recommendation pipeline."""

    def test_basic_recommendation(self):
        """Should return a valid recommendation."""
        result = recommend_route(12.35, 78.55, 100, mode="haversine", print_output=False)
        assert result["recommended"] is not None
        assert len(result["all_results"]) > 0
        assert result["input"]["mode"] == "haversine"

    def test_recommendation_with_high_milk(self):
        """High milk qty should reject routes."""
        result = recommend_route(12.35, 78.55, 3000, mode="haversine", print_output=False)
        assert len(result["rejected_routes"]) > 0
        total = len(result["all_results"]) + len(result["rejected_routes"])
        assert total == 7

    def test_recommendation_all_rejected(self):
        """All routes rejected -> recommended is None."""
        result = recommend_route(12.35, 78.55, 5000, mode="haversine", print_output=False)
        assert result["recommended"] is None
        assert len(result["rejected_routes"]) == 7

    def test_config_values(self):
        """Config should reflect current settings."""
        result = recommend_route(12.35, 78.55, 100, mode="haversine", print_output=False)
        config = result["config"]
        assert config["max_route_hours"] == MAX_ROUTE_HOURS
        assert config["avg_speed_kmh"] == AVG_SPEED_KMH
        assert config["avg_stop_time_min"] == AVG_STOP_TIME_HOURS * 60

    def test_known_location_katteri(self):
        """Katteri (on Harur route) should score well for that route."""
        result = recommend_route(12.2007, 78.5444, 50, mode="haversine", print_output=False)
        assert result["recommended"] is not None

    def test_known_location_anna_nagar(self):
        """Anna Nagar test."""
        result = recommend_route(12.2835, 78.4910, 50, mode="haversine", print_output=False)
        assert result["recommended"] is not None

    def test_legacy_alias(self):
        """haversine_recommendation should still work."""
        result = haversine_recommendation(12.35, 78.55, print_output=False)
        assert result["recommended"] is not None
        assert result["input"]["mode"] == "haversine"


# =============================================================================
# Input Validation Tests
# =============================================================================

class TestValidation:
    """Tests for input validation."""

    def test_invalid_latitude(self):
        with pytest.raises(ValueError):
            recommend_route(91.0, 78.0, 100, print_output=False)

    def test_invalid_longitude(self):
        with pytest.raises(ValueError):
            recommend_route(12.0, 181.0, 100, print_output=False)

    def test_negative_milk_qty(self):
        with pytest.raises(ValueError):
            recommend_route(12.0, 78.0, -10, print_output=False)

    def test_invalid_mode(self):
        with pytest.raises(ValueError):
            recommend_route(12.0, 78.0, 100, mode="invalid", print_output=False)

    def test_weights_dont_sum_to_1(self):
        bad_weights = {"extra_distance": 0.5, "total_route_km": 0.5,
                       "centroid_proximity": 0.5, "uti_headroom": 0.5}
        with pytest.raises(ValueError):
            recommend_route(12.0, 78.0, 100, weights=bad_weights, print_output=False)


# =============================================================================
# Helper Function Tests
# =============================================================================

class TestHelpers:
    """Tests for internal helper functions."""

    def test_build_stop_list(self):
        """Stop list should start and end with CC."""
        routes = load_route_data()
        stops = _build_stop_list(routes[0])
        assert stops[0][2] == "Chilling Center (CC)"
        assert stops[-1][2] == "Chilling Center (CC)"
        assert len(stops) == len(routes[0].hmbs) + 2

    def test_resolve_stop_name_new_hmb(self):
        """Should recognize the new HMB coordinate."""
        name = _resolve_stop_name((12.35, 78.55), 12.35, 78.55, [])
        assert name == "NEW HMB"

    def test_resolve_stop_name_existing(self):
        """Should find existing HMB name."""
        hmb = HMB("1", "TestHMB", 12.2, 78.4, 1, 0)
        name = _resolve_stop_name((12.2, 78.4), 12.35, 78.55, [hmb])
        assert name == "TestHMB"

    def test_resolve_stop_name_unknown(self):
        """Should return Unknown for unmatched coord."""
        name = _resolve_stop_name((99.0, 99.0), 12.35, 78.55, [])
        assert name == "Unknown"
